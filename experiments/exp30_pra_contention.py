"""Proactive Rank Adaptation (PRA) under time-varying CPU contention.

This is the result experiment for PRA. It drives the anytime CP+S-AFF model
(exp29) with a synthetic but reproducible CPU-availability trace and compares
rank-selection policies:

  fixed_r2   : always cheapest rank (never misses, low accuracy)
  fixed_r8   : always most accurate rank (misses deadlines under load)
  reactive   : lower rank one step after a miss, raise after sustained slack
  proactive  : forecast next-window CPU (EWMA) and pick the best feasible rank
               ahead of the deadline, with hysteresis (Algorithm 1 in paper)
  oracle     : knows the true CPU share for the frame (upper bound)

Latency model L(R) is consistent with exp28: CP extraction cost scales linearly
with rank, plus a small S-AFF term. Per-frame accuracy pck20_by_rank comes from
the *measured* anytime model (exp29 dump), so accuracy is real; only the
contention trace is simulated.

Metrics per policy:
  deadline_miss_rate  : fraction of frames whose effective latency exceeds D
  eff_pck20           : delivered PCK20 counting a missed frame as 0 (dropped)
  ontime_pck20        : mean PCK20 over frames that met the deadline
  switches            : number of rank changes
  mean_rank           : average selected rank
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def latency_model(ranks, rank4_cp_us, cp_iters, saff_us_map):
    """Total single-core (100% CPU) latency per rank, in microseconds."""
    L = {}
    for r in ranks:
        cp_us = rank4_cp_us * (r / 4.0) * (cp_iters / 10.0)
        L[r] = cp_us + saff_us_map.get(r, 180.0)
    return L


def make_cpu_trace(n, seed, min_rho=0.12, mean_regime=250):
    """Piecewise-regime CPU-availability trace in (0,1] with within-regime noise."""
    rng = np.random.RandomState(seed)
    rho = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        length = max(20, int(rng.exponential(mean_regime)))
        base = float(rng.choice([0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0]))
        seg = base + rng.normal(0.0, 0.05, size=min(length, n - i))
        rho[i : i + len(seg)] = seg
        i += len(seg)
    return np.clip(rho, min_rho, 1.0)


def feasible_max_rank(L, ranks, rho, D):
    feas = [r for r in ranks if L[r] / rho <= D]
    return max(feas) if feas else min(ranks)


def run_policy(policy, L, ranks, rho_trace, D, pck20_by_rank, rank_idx, alpha=0.3, delta=0.0,
               hysteresis_frames=5, safety_margin=0.15):
    n = len(rho_trace)
    chosen = np.empty(n, dtype=np.int64)
    R = 4
    rho_hat = 1.0
    slack_run = 0
    up_run = 0
    D_plan = D * (1.0 - safety_margin)  # risk-aware: plan against a tightened deadline
    for t in range(n):
        rho_true = rho_trace[t]
        if policy == "fixed_r2":
            R = min(ranks)
        elif policy == "fixed_r8":
            R = max(ranks)
        elif policy == "oracle":
            R = feasible_max_rank(L, ranks, rho_true, D)
        elif policy == "proactive":
            # proactive: forecast next-window CPU share, pick best feasible rank ahead
            # of the deadline. Drop immediately to protect the deadline; raise only
            # after the forecast has supported a higher rank for several frames.
            rho_hat = alpha * (rho_trace[t - 1] if t > 0 else rho_true) + (1 - alpha) * rho_hat
            cand = feasible_max_rank(L, ranks, rho_hat, D_plan)
            if cand < R or L[R] / rho_hat > D_plan:
                R = cand
                up_run = 0
            elif cand > R:
                up_run += 1
                if up_run >= hysteresis_frames:
                    R = cand
                    up_run = 0
            else:
                up_run = 0
        elif policy == "reactive":
            if t > 0:
                prev_lat = L[R] / rho_trace[t - 1]
                if prev_lat > D:  # missed -> step down
                    lower = [r for r in ranks if r < R]
                    if lower:
                        R = max(lower)
                    slack_run = 0
                else:
                    slack_run += 1
                    if slack_run >= hysteresis_frames:  # sustained slack -> step up
                        higher = [r for r in ranks if r > R]
                        if higher:
                            R = min(higher)
                        slack_run = 0
        else:
            raise ValueError(policy)
        chosen[t] = R

    eff_lat = np.array([L[chosen[t]] / rho_trace[t] for t in range(n)])
    miss = eff_lat > D
    acc = np.array([pck20_by_rank[t, rank_idx[chosen[t]]] for t in range(n)])
    switches = int(np.sum(chosen[1:] != chosen[:-1]))
    return {
        "policy": policy,
        "deadline_miss_rate": float(miss.mean()),
        "eff_pck20": float((acc * (~miss)).mean()),
        "ontime_pck20": float(acc[~miss].mean()) if (~miss).any() else 0.0,
        "mean_rank": float(chosen.mean()),
        "switches": switches,
        "mean_eff_latency_ms": float(eff_lat.mean() / 1000.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-eval", type=Path, default=Path("outputs/anytime_pra_rank_eval.npz"))
    parser.add_argument("--csv", type=Path, default=Path("results/pra_contention_policies.csv"))
    parser.add_argument("--plot", type=Path, default=Path("PAPER/figures/fig_pra_contention.png"))
    parser.add_argument("--deadline-ms", type=float, default=20.0)
    parser.add_argument("--rank4-cp-us", type=float, default=3400.0)
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--hysteresis-frames", type=int, default=5)
    parser.add_argument("--trace-seed", type=int, default=13)
    parser.add_argument("--trace-len", type=int, default=0, help="0 = use all test frames")
    args = parser.parse_args()

    z = np.load(args.rank_eval)
    ranks = [int(r) for r in z["ranks"]]
    pck20_by_rank = z["pck20_by_rank"]
    rank_idx = {r: i for i, r in enumerate(ranks)}

    n = pck20_by_rank.shape[0] if args.trace_len == 0 else min(args.trace_len, pck20_by_rank.shape[0])
    pck20_by_rank = pck20_by_rank[:n]

    saff_us_map = {2: 172.0, 4: 204.0, 6: 156.0, 8: 200.0}  # from exp28 ONNX profiling
    L = latency_model(ranks, args.rank4_cp_us, args.cp_iters, saff_us_map)
    D = args.deadline_ms * 1000.0
    rho_trace = make_cpu_trace(n, args.trace_seed)

    print("Latency model L(R) at 100% CPU (ms):", {r: round(L[r] / 1000.0, 3) for r in ranks})
    print("Measured A(R) PCK20:", {r: round(float(pck20_by_rank[:, rank_idx[r]].mean()), 2) for r in ranks})
    print(f"Deadline={args.deadline_ms}ms  frames={n}  mean_cpu={rho_trace.mean():.2f}", flush=True)

    policies = ["fixed_r2", "fixed_r8", "reactive", "proactive", "oracle"]
    rows = []
    for p in policies:
        res = run_policy(
            p, L, ranks, rho_trace, D, pck20_by_rank, rank_idx,
            alpha=args.alpha, hysteresis_frames=args.hysteresis_frames,
            safety_margin=args.safety_margin,
        )
        rows.append(res)
        print(res, flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved_csv={args.csv}")

    # optional bar plot: eff_pck20 and miss rate per policy
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["policy"] for r in rows]
        eff = [r["eff_pck20"] for r in rows]
        miss = [100.0 * r["deadline_miss_rate"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(names))
        ax1.bar(x - 0.2, eff, width=0.4, color="#2b7bba", label="Delivered PCK20")
        ax1.set_ylabel("Delivered PCK$_{20}$ (miss=0)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15)
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, miss, width=0.4, color="#d1495b", label="Deadline miss %")
        ax2.set_ylabel("Deadline miss rate (%)")
        ax1.set_title(f"PRA under CPU contention (D={args.deadline_ms} ms)")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"saved_plot={args.plot}")
    except Exception as exc:  # pragma: no cover
        print(f"plot_skipped={exc}")

    print("SUMMARY " + json.dumps({r["policy"]: {"eff_pck20": r["eff_pck20"], "miss": r["deadline_miss_rate"]} for r in rows}))


if __name__ == "__main__":
    main()
