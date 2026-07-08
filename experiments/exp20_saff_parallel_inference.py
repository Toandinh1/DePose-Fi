"""Benchmark branch-level parallelism for S-AFF inference.

This script tests whether the S-AFF architecture exposes useful hardware
parallelism before we try it on a Raspberry Pi or Jetson. It uses synthetic CP
features with the same dimensions as our Person-in-WiFi 3D dual-CP model.

The benchmark compares:
  - sequential execution,
  - parallel A/B/C component branches inside each stream,
  - parallel amplitude/phase streams,
  - both branch-level and stream-level parallelism.

It also sweeps torch CPU thread counts to emulate small edge CPUs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn


LINKS = 9
SUBCARRIERS = 30
PACKETS = 20
JOINTS = 14
MAX_PEOPLE = 3


def saff_dims(model_size: str):
    if model_size == "small":
        return 32, 64, 32, 64, 32, 16
    if model_size == "medium":
        return 64, 128, 64, 128, 64, 32
    if model_size == "large":
        return 96, 256, 96, 256, 96, 48
    raise ValueError(f"Unknown model_size={model_size}")


class ParallelFactorEncoder(nn.Module):
    def __init__(self, rank: int, model_size: str, temperature: float = 0.7):
        super().__init__()
        self.rank = rank
        self.temperature = temperature
        a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = saff_dims(model_size)
        self.a_size = LINKS * rank
        self.b_size = SUBCARRIERS * rank
        self.hidden = f_dim

        self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(rank * LINKS, a_dim), nn.ReLU())
        self.b_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(rank, rank), nn.Sigmoid())
        self.b_net = nn.Sequential(
            nn.Conv1d(rank, b_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(b_channels, b_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(6),
            nn.Flatten(),
            nn.Linear(b_channels * 6, b_dim),
            nn.ReLU(),
        )
        self.c_net = nn.Sequential(
            nn.Conv1d(rank, c_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(c_channels * 4, c_dim),
            nn.ReLU(),
        )
        total_dim = a_dim + b_dim + c_dim
        self.fuse_net = nn.Sequential(nn.Linear(total_dim, f_dim), nn.ReLU())
        self.gate = nn.Linear(total_dim, 4)
        self.proj = nn.ModuleList(
            [nn.Linear(a_dim, f_dim), nn.Linear(b_dim, f_dim), nn.Linear(c_dim, f_dim), nn.Linear(f_dim, f_dim)]
        )

    def split(self, x: torch.Tensor):
        a = x[:, : self.a_size].reshape((-1, self.rank, LINKS))
        b = x[:, self.a_size : self.a_size + self.b_size].reshape((-1, self.rank, SUBCARRIERS))
        c = x[:, self.a_size + self.b_size :].reshape((-1, self.rank, PACKETS))
        return a, b, c

    def encode_a(self, a: torch.Tensor):
        return self.a_net(a)

    def encode_b(self, b: torch.Tensor):
        return self.b_net(b * self.b_att(b).unsqueeze(-1))

    def encode_c(self, c: torch.Tensor):
        return self.c_net(c)

    def finish(self, fa: torch.Tensor, fb: torch.Tensor, fc: torch.Tensor):
        h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
        ff = self.fuse_net(h)
        gates = torch.softmax(self.gate(h) / self.temperature, dim=1)
        branches = torch.stack([self.proj[0](fa), self.proj[1](fb), self.proj[2](fc), self.proj[3](ff)], dim=1)
        return torch.sum(gates.unsqueeze(-1) * branches, dim=1)

    def forward_sequential(self, x: torch.Tensor):
        a, b, c = self.split(x)
        return self.finish(self.encode_a(a), self.encode_b(b), self.encode_c(c))

    def forward_branch_parallel(self, x: torch.Tensor, executor: ThreadPoolExecutor):
        a, b, c = self.split(x)
        fa_f = executor.submit(self.encode_a, a)
        fb_f = executor.submit(self.encode_b, b)
        fc_f = executor.submit(self.encode_c, c)
        return self.finish(fa_f.result(), fb_f.result(), fc_f.result())


class ParallelDualSaff(nn.Module):
    def __init__(self, rank: int, model_size: str, num_queries: int, temperature: float = 0.7):
        super().__init__()
        _, _, _, hidden, _, _ = saff_dims(model_size)
        self.amp_encoder = ParallelFactorEncoder(rank, model_size, temperature)
        self.phase_encoder = ParallelFactorEncoder(rank, model_size, temperature)
        self.cross_gate = nn.Sequential(nn.Linear(hidden * 2, 2), nn.Softmax(dim=1))
        self.cross_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
        self.query_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pose_head = nn.Linear(hidden, JOINTS * 3)
        self.cls_head = nn.Linear(hidden, 1)
        self.count_head = nn.Linear(hidden, MAX_PEOPLE + 1)

    def finish(self, ha: torch.Tensor, hp: torch.Tensor):
        both = torch.cat([ha, hp], dim=1)
        stream_gate = self.cross_gate(both)
        fused = self.cross_fuse(both) + stream_gate[:, 0:1] * ha + stream_gate[:, 1:2] * hp
        q = fused.unsqueeze(1) + self.query_embed.unsqueeze(0)
        q = self.query_net(q)
        poses = self.pose_head(q).reshape((q.shape[0], q.shape[1], JOINTS, 3))
        logits = self.cls_head(q).squeeze(-1)
        count_logits = self.count_head(fused)
        return poses, logits, count_logits

    def forward_sequential(self, xa: torch.Tensor, xp: torch.Tensor):
        ha = self.amp_encoder.forward_sequential(xa)
        hp = self.phase_encoder.forward_sequential(xp)
        return self.finish(ha, hp)

    def forward_branch_parallel(self, xa: torch.Tensor, xp: torch.Tensor, executor: ThreadPoolExecutor):
        ha = self.amp_encoder.forward_branch_parallel(xa, executor)
        hp = self.phase_encoder.forward_branch_parallel(xp, executor)
        return self.finish(ha, hp)

    def forward_stream_parallel(self, xa: torch.Tensor, xp: torch.Tensor, executor: ThreadPoolExecutor):
        ha_f = executor.submit(self.amp_encoder.forward_sequential, xa)
        hp_f = executor.submit(self.phase_encoder.forward_sequential, xp)
        return self.finish(ha_f.result(), hp_f.result())

    def forward_full_parallel(self, xa: torch.Tensor, xp: torch.Tensor, executor: ThreadPoolExecutor):
        ha_f = executor.submit(self.amp_encoder.forward_branch_parallel, xa, executor)
        hp_f = executor.submit(self.phase_encoder.forward_branch_parallel, xp, executor)
        return self.finish(ha_f.result(), hp_f.result())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def benchmark(fn, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        lat_us = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            lat_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return {
        "latency_mean_us": statistics.mean(lat_us),
        "latency_median_us": statistics.median(lat_us),
        "latency_p95_us": percentile(lat_us, 95),
        "latency_min_us": min(lat_us),
    }


def run(args):
    out_rows = []
    feature_dim = args.rank * (LINKS + SUBCARRIERS + PACKETS)
    threads_to_test = [int(x) for x in args.torch_threads.split(",")]
    torch.set_num_interop_threads(args.interop_threads)

    for torch_threads in threads_to_test:
        torch.set_num_threads(torch_threads)

        for model_size in args.model_sizes.split(","):
            torch.manual_seed(args.seed)
            model = ParallelDualSaff(args.rank, model_size, args.num_queries, args.temperature).eval()
            xa = torch.randn(args.batch_size, feature_dim)
            xp = torch.randn(args.batch_size, feature_dim)
            params = count_params(model)

            modes = [
                ("sequential", lambda executor: lambda: model.forward_sequential(xa, xp), 1),
                ("branch_parallel", lambda executor: lambda: model.forward_branch_parallel(xa, xp, executor), 6),
                ("stream_parallel", lambda executor: lambda: model.forward_stream_parallel(xa, xp, executor), 2),
                ("full_parallel", lambda executor: lambda: model.forward_full_parallel(xa, xp, executor), 8),
            ]
            baseline_mean = None
            for mode_name, make_fn, workers in modes:
                max_workers = max(workers, args.parallel_workers)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    stats = benchmark(make_fn(executor), args.warmup, args.iters)
                if mode_name == "sequential":
                    baseline_mean = stats["latency_mean_us"]
                row = {
                    "model": "dual_cp_saff",
                    "model_size": model_size,
                    "rank": args.rank,
                    "batch_size": args.batch_size,
                    "params": params,
                    "torch_threads": torch_threads,
                    "interop_threads": args.interop_threads,
                    "mode": mode_name,
                    "speedup_vs_sequential": baseline_mean / stats["latency_mean_us"] if baseline_mean else 1.0,
                    **stats,
                }
                out_rows.append(row)
                print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"saved_csv={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--model-sizes", default="small,medium,large")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", default="1,2,4")
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/saff_parallel_inference.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
