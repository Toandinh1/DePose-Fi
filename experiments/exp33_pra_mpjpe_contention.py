"""PRA contention simulation for single-person dual-CP anytime model, MPJPE metric.

Companion to exp32. Drives the anytime dual-CP+S-AFF model under a time-varying
CPU-availability trace and compares rank-selection policies, scoring delivered
pose quality in MPJPE (mm). Unlike the PCK sim (exp30), a missed-deadline frame
cannot be scored 0 (that would be perfect); instead a missed frame is charged the
per-frame mean-pose baseline error (i.e. "no valid pose delivered this slot").

Latency L(R): dual-CP does TWO CP extractions (amp + phase) per frame, each cost
scaling ~linearly with rank, plus a small S-AFF term. Real per-rank CP cost can
be supplied via --latency-json (from exp34 calibration); otherwise a linear model
is used. Accuracy (per-frame MPJPE by rank) is measured (exp32 dump).

Reported per policy:
  drop_rate        : fraction of frames whose effective latency exceeds D
  eff_mpjpe_mm     : delivered MPJPE, missed frame charged mean-pose baseline
  ontime_mpjpe_mm  : MPJPE over frames that met the deadline
  mean_rank, switches, mean_eff_latency_ms
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def latency_model(ranks, cp_us_at_rank4, cp_iters, saff_us, latency_json=None, cp_count=2):
    """Total per-frame latency at 100% CPU (us). cp_count extractions per frame
    (dual-CP amp+phase => 2; single-CP sanitized-phase => 1)."""
    if latency_json and Path(latency_json).exists():
        meas = {int(k): float(v) for k, v in json.loads(Path(latency_json).read_text()).items()}
        return {r: meas[r] for r in ranks}
    L = {}
    for r in ranks:
        cp_us = cp_count * cp_us_at_rank4 * (r / 4.0) * (cp_iters / 10.0)
        L[r] = cp_us + saff_us
    return L


def make_cpu_trace(n, seed, min_rho=0.12, mean_regime=250):
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
    """Highest rank whose effective latency fits the deadline (best affordable accuracy)."""
    feas = [r for r in ranks if L[r] / rho <= D]
    return max(feas) if feas else min(ranks)


def run_policy(policy, L, ranks, rho_trace, D, mpjpe_by_rank, mean_pf, rank_idx,
               alpha=0.3, hysteresis_frames=5, safety_margin=0.15):
    n = len(rho_trace)
    chosen = np.empty(n, dtype=np.int64)
    R = min(ranks)
    rho_hat = 1.0
    slack_run = 0
    up_run = 0
    D_plan = D * (1.0 - safety_margin)
    for t in range(n):
        rho_true = rho_trace[t]
        if policy == "fixed_min":
            R = min(ranks)
        elif policy == "fixed_max":
            R = max(ranks)
        elif policy == "oracle":
            R = feasible_max_rank(L, ranks, rho_true, D)
        elif policy == "proactive":
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
                if L[R] / rho_trace[t - 1] > D:
                    lower = [r for r in ranks if r < R]
                    if lower:
                        R = max(lower)
                    slack_run = 0
                else:
                    slack_run += 1
                    if slack_run >= hysteresis_frames:
                        higher = [r for r in ranks if r > R]
                        if higher:
                            R = min(higher)
                        slack_run = 0
        else:
            raise ValueError(policy)
        chosen[t] = R

    eff_lat = np.array([L[chosen[t]] / rho_trace[t] for t in range(n)])
    miss = eff_lat > D
    ontime_err = np.array([mpjpe_by_rank[t, rank_idx[chosen[t]]] for t in range(n)])
    delivered = np.where(miss, mean_pf, ontime_err)  # missed -> mean-pose baseline cost
    switches = int(np.sum(chosen[1:] != chosen[:-1]))
    return {
        "policy": policy,
        "drop_rate": float(miss.mean()),
        "eff_mpjpe_mm": float(delivered.mean()),
        "ontime_mpjpe_mm": float(ontime_err[~miss].mean()) if (~miss).any() else float("nan"),
        "mean_rank": float(chosen.mean()),
        "switches": switches,
        "mean_eff_latency_ms": float(eff_lat.mean() / 1000.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-eval", type=Path,
                        default=Path("outputs/piw_dualcp_anytime_rank_eval.npz"))
    parser.add_argument("--csv", type=Path, default=Path("results/piw_dualcp_pra_contention.csv"))
    parser.add_argument("--plot", type=Path, default=Path("PAPER/figures/fig_piw_dualcp_pra.png"))
    parser.add_argument("--deadline-ms", type=float, default=33.0)
    parser.add_argument("--cp-us-at-rank4", type=float, default=1500.0, help="single-CP cost at rank-4 (us)")
    parser.add_argument("--cp-count", type=int, default=2, help="CP extractions per frame (dual=2, single=1)")
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--saff-us", type=float, default=250.0)
    parser.add_argument("--latency-json", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--hysteresis-frames", type=int, default=5)
    parser.add_argument("--trace-seed", type=int, default=13)
    args = parser.parse_args()

    z = np.load(args.rank_eval)
    ranks = [int(r) for r in z["ranks"]]
    mpjpe_by_rank = z["mpjpe_by_rank_mm"]
    mean_pf = z["mean_pose_by_frame_mm"]
    rank_idx = {r: i for i, r in enumerate(ranks)}
    n = mpjpe_by_rank.shape[0]

    L = latency_model(ranks, args.cp_us_at_rank4, args.cp_iters, args.saff_us, args.latency_json, args.cp_count)
    D = args.deadline_ms * 1000.0
    rho_trace = make_cpu_trace(n, args.trace_seed)

    print("L(R) @100% CPU (ms):", {r: round(L[r] / 1000.0, 3) for r in ranks})
    print("A(R) MPJPE (mm):", {r: round(float(mpjpe_by_rank[:, rank_idx[r]].mean()), 2) for r in ranks})
    print(f"mean_pose_baseline_mm={mean_pf.mean():.1f}  D={args.deadline_ms}ms  frames={n}  mean_cpu={rho_trace.mean():.2f}", flush=True)

    policies = ["fixed_min", "fixed_max", "reactive", "proactive", "oracle"]
    rows = []
    for p in policies:
        res = run_policy(p, L, ranks, rho_trace, D, mpjpe_by_rank, mean_pf, rank_idx,
                         alpha=args.alpha, hysteresis_frames=args.hysteresis_frames,
                         safety_margin=args.safety_margin)
        rows.append(res)
        print(res, flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved_csv={args.csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = {"fixed_min": "#7f7f7f", "fixed_max": "#d1495b", "reactive": "#edae49",
                  "proactive": "#2b7bba", "oracle": "#2a9d8f"}
        for r in rows:
            ax.scatter(100.0 * r["drop_rate"], r["eff_mpjpe_mm"], s=120,
                       color=colors.get(r["policy"], "#333"), zorder=3, label=r["policy"])
            ax.annotate(r["policy"], (100.0 * r["drop_rate"], r["eff_mpjpe_mm"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_xlabel("Deadline drop rate (%)")
        ax.set_ylabel("Effective MPJPE (mm, lower=better)")
        ax.set_title(f"PRA on single-person dual-CP (D={args.deadline_ms} ms)\nbest = bottom-left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"saved_plot={args.plot}")
    except Exception as exc:  # pragma: no cover
        print(f"plot_skipped={exc}")

    print("SUMMARY " + json.dumps({r["policy"]: {"drop": r["drop_rate"], "eff_mpjpe": r["eff_mpjpe_mm"]} for r in rows}))


if __name__ == "__main__":
    main()
