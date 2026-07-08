"""Runtime benchmark for hard-routed MM-Fi S-AFF subcarrier-only execution.

exp24 shows the trained S-AFF gate collapses to the subcarrier expert on the
MM-Fi split. This script measures the runtime of executing only that branch.

Weights are random here because latency depends on operator graph shape, not
trained parameter values. Accuracy comes from exp24_hard_routed_saff.py.
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


RANK = 4
SUBCARRIERS = 114
OUT_DIM = 17 * 3


class FixedAdaptiveAvgPool1d(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.bins = []
        for i in range(output_size):
            start = int((i * input_size) // output_size)
            end = int(((i + 1) * input_size + output_size - 1) // output_size)
            self.bins.append((start, max(end, start + 1)))

    def forward(self, x):
        return torch.cat([x[:, :, start:end].mean(dim=2, keepdim=True) for start, end in self.bins], dim=2)


class MMFiSubcarrierExpert(nn.Module):
    def __init__(self, exportable_pool: bool = False):
        super().__init__()
        pool = FixedAdaptiveAvgPool1d(SUBCARRIERS, 8) if exportable_pool else nn.AdaptiveAvgPool1d(8)
        self.b_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(RANK, RANK), nn.Sigmoid())
        self.b_net = nn.Sequential(
            nn.Conv1d(RANK, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            pool,
            nn.Flatten(),
            nn.Linear(32 * 8, 96),
            nn.ReLU(),
        )
        self.head = nn.Linear(96, OUT_DIM)

    def forward(self, b):
        fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
        return self.head(fb)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def benchmark(fn, warmup, iters):
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


def export_onnx(model, b, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (b,),
            str(path),
            input_names=["b"],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--intra-threads", default="1,2,4")
    parser.add_argument("--inter-threads", default="1,2")
    parser.add_argument("--execution-modes", default="sequential,parallel")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/onnx_mmfi_bonly"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mmfi_bonly_runtime.csv"))
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(7)
    model = MMFiSubcarrierExpert(exportable_pool=False).eval()
    export_model = MMFiSubcarrierExpert(exportable_pool=True).eval()
    export_model.load_state_dict(model.state_dict(), strict=True)
    b = torch.randn(args.batch_size, RANK, SUBCARRIERS)
    b_np = b.numpy()
    params = count_params(model)

    rows = []
    torch_stats = benchmark(lambda: model(b), args.warmup, args.iters)
    rows.append(
        {
            "model": "mmfi_saff_subcarrier_only",
            "mode": "torch",
            "params": params,
            "batch_size": args.batch_size,
            "intra_threads": "",
            "inter_threads": "",
            "execution_mode": "",
            "speedup_vs_torch": 1.0,
            **torch_stats,
        }
    )
    print(rows[-1])

    onnx_path = args.export_dir / "mmfi_saff_subcarrier_only.onnx"
    if args.rebuild or not onnx_path.exists():
        export_onnx(export_model, b, onnx_path)

    for intra in [int(v) for v in args.intra_threads.split(",")]:
        for inter in [int(v) for v in args.inter_threads.split(",")]:
            for execution_mode in args.execution_modes.split(","):
                sess = make_session(onnx_path, intra, inter, execution_mode)
                stats = benchmark(lambda: sess.run(None, {"b": b_np}), args.warmup, args.iters)
                rows.append(
                    {
                        "model": "mmfi_saff_subcarrier_only",
                        "mode": "onnx",
                        "params": params,
                        "batch_size": args.batch_size,
                        "intra_threads": intra,
                        "inter_threads": inter,
                        "execution_mode": execution_mode,
                        "speedup_vs_torch": torch_stats["latency_mean_us"] / stats["latency_mean_us"],
                        **stats,
                    }
                )
                print(rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={args.output}")
    print(f"onnx_path={onnx_path}")


if __name__ == "__main__":
    main()
