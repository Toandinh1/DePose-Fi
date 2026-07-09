"""Resource-aware CP-rank selection simulation.

This experiment tests the systems idea:

  available CPU budget -> choose the highest CP-rank profile that satisfies
  the latency target.

The script measures ONNX Runtime latency for rank-specific S-AFF graphs and
combines it with a calibrated CP extraction latency model. Accuracy values for
R=2/6/8 are explicit scenario assumptions until those exact rank profiles are
trained; R=4 can be anchored to the measured MM-Fi CP+S-AFF result.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install with: python -m pip install onnx onnxruntime") from exc


LINKS = 3
SUBCARRIERS = 114
PACKETS = 10
OUT_DIM = 17 * 3


class FixedAdaptiveAvgPool1d(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        bins = []
        for i in range(output_size):
            start = int((i * input_size) // output_size)
            end = int(((i + 1) * input_size + output_size - 1) // output_size)
            bins.append((start, max(end, start + 1)))
        self.bins = bins

    def forward(self, x):
        return torch.cat([x[:, :, start:end].mean(dim=2, keepdim=True) for start, end in self.bins], dim=2)


class RankSaff(nn.Module):
    def __init__(self, rank: int, temperature: float = 0.7):
        super().__init__()
        self.rank = rank
        self.temperature = temperature
        self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(rank * LINKS, 32), nn.ReLU())
        self.b_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(rank, rank), nn.Sigmoid())
        self.b_net = nn.Sequential(
            nn.Conv1d(rank, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            FixedAdaptiveAvgPool1d(SUBCARRIERS, 8),
            nn.Flatten(),
            nn.Linear(32 * 8, 96),
            nn.ReLU(),
        )
        self.c_net = nn.Sequential(
            nn.Conv1d(rank, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            FixedAdaptiveAvgPool1d(PACKETS, 4),
            nn.Flatten(),
            nn.Linear(16 * 4, 32),
            nn.ReLU(),
        )
        self.fuse_net = nn.Sequential(nn.Linear(32 + 96 + 32, 96), nn.ReLU())
        self.gate = nn.Linear(32 + 96 + 32, 4)
        self.heads = nn.ModuleList(
            [
                nn.Linear(32, OUT_DIM),
                nn.Linear(96, OUT_DIM),
                nn.Linear(32, OUT_DIM),
                nn.Linear(96, OUT_DIM),
            ]
        )

    def forward(self, x):
        z = x[:, 0]
        a = z[:, :, :LINKS]
        b = z[:, :, LINKS : LINKS + SUBCARRIERS]
        c = z[:, :, LINKS + SUBCARRIERS :]
        fa = self.a_net(a)
        fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
        fc = self.c_net(c)
        h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
        ff = self.fuse_net(h)
        gates = torch.softmax(self.gate(h) / self.temperature, dim=1)
        preds = torch.stack(
            [
                self.heads[0](fa),
                self.heads[1](fb),
                self.heads[2](fc),
                self.heads[3](ff),
            ],
            dim=1,
        )
        return torch.sum(gates.unsqueeze(-1) * preds, dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def benchmark(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    lat_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        lat_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return {
        "onnx_mean_us": statistics.mean(lat_us),
        "onnx_median_us": statistics.median(lat_us),
        "onnx_p95_us": percentile(lat_us, 95),
        "onnx_min_us": min(lat_us),
    }


def export_onnx(model, x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (x,),
            str(path),
            input_names=["x"],
            output_names=["pose"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))


def make_session(path, intra_threads, inter_threads, execution_mode):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL if execution_mode == "parallel" else ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def parse_float_map(spec: str):
    out = {}
    for part in spec.split(","):
        key, value = part.split(":")
        out[int(key.strip())] = float(value.strip())
    return out


def choose_profile(profile_rows, availability, target_latency_us):
    feasible = [row for row in profile_rows if row["effective_total_us"] <= target_latency_us]
    if feasible:
        return max(feasible, key=lambda row: (row["assumed_pck20"], row["rank"]))
    return min(profile_rows, key=lambda row: row["effective_total_us"])


def run(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    ranks = [int(v) for v in args.ranks.split(",")]
    assumed_pck = parse_float_map(args.assumed_pck20)
    availabilities = [float(v) for v in args.cpu_availability.split(",")]

    profile_rows = []
    for rank in ranks:
        model = RankSaff(rank, args.temperature).eval()
        x = torch.randn(args.batch_size, 1, rank, LINKS + SUBCARRIERS + PACKETS, dtype=torch.float32)
        x_np = x.numpy()
        onnx_path = args.export_dir / f"rank{rank}" / "mmfi_saff.onnx"
        if args.rebuild or not onnx_path.exists():
            export_onnx(model, x, onnx_path)

        session = make_session(onnx_path, args.intra_threads, args.inter_threads, args.execution_mode)
        stats = benchmark(lambda: session.run(None, {"x": x_np}), args.warmup, args.iters)
        cp_us = args.rank4_cp_us * (rank / 4.0) * (args.cp_iters / 10.0)
        total_us = cp_us + stats["onnx_mean_us"]
        row = {
            "rank": rank,
            "cp_iters": args.cp_iters,
            "params": count_params(model),
            "assumed_pck20": assumed_pck[rank],
            "cp_latency_us_at_100pct": cp_us,
            "onnx_latency_us_at_100pct": stats["onnx_mean_us"],
            "total_latency_us_at_100pct": total_us,
            "fps_at_100pct": 1_000_000.0 / total_us,
            **stats,
        }
        profile_rows.append(row)
        print(row, flush=True)

    scenario_rows = []
    for availability in availabilities:
        candidates = []
        for row in profile_rows:
            effective = row["total_latency_us_at_100pct"] / availability
            candidates.append(
                {
                    **row,
                    "cpu_availability": availability,
                    "effective_total_us": effective,
                    "effective_fps": 1_000_000.0 / effective,
                    "target_latency_us": args.target_latency_us,
                    "meets_target": effective <= args.target_latency_us,
                }
            )
        chosen = choose_profile(candidates, availability, args.target_latency_us)
        for row in candidates:
            scenario_rows.append({**row, "selected": row["rank"] == chosen["rank"], "selected_rank": chosen["rank"]})
        print(
            f"availability={availability:.2f} selected_rank={chosen['rank']} "
            f"effective_ms={chosen['effective_total_us'] / 1000.0:.2f} "
            f"assumed_pck20={chosen['assumed_pck20']:.2f}",
            flush=True,
        )

    args.output_profiles.parent.mkdir(parents=True, exist_ok=True)
    with args.output_profiles.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    args.output_scenarios.parent.mkdir(parents=True, exist_ok=True)
    with args.output_scenarios.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)
    print(f"saved_profiles={args.output_profiles}")
    print(f"saved_scenarios={args.output_scenarios}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="2,4,6,8")
    parser.add_argument("--assumed-pck20", default="2:47.0,4:50.8,6:51.6,8:52.1")
    parser.add_argument("--cpu-availability", default="0.2,0.4,0.8")
    parser.add_argument("--target-latency-us", type=float, default=20_000.0)
    parser.add_argument("--rank4-cp-us", type=float, default=3400.0)
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--intra-threads", type=int, default=1)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--execution-mode", choices=["sequential", "parallel"], default="sequential")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/onnx_resource_rank_selector"))
    parser.add_argument("--output-profiles", type=Path, default=Path("outputs/resource_rank_profiles.csv"))
    parser.add_argument("--output-scenarios", type=Path, default=Path("outputs/resource_rank_scenarios.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
