"""Benchmark MM-Fi CP + S-AFF runtime.

This is the fair runtime counterpart to HPE-Li on MM-Fi. It uses the same
CPSelectiveAFF architecture from exp14 with input shape N x 1 x R x 127.
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install with: python -m pip install onnx onnxruntime") from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp14_cp_cnn_aff import CPSelectiveAFF, OUT_DIM, RANK  # noqa: E402


class FixedAdaptiveAvgPool1d(nn.Module):
    """Static-shape replacement for AdaptiveAvgPool1d that exports to ONNX.

    PyTorch's legacy ONNX exporter rejects adaptive average pooling when the
    output size is not an even factor of the input size. MM-Fi S-AFF uses fixed
    lengths, so we can express the same bins with slice + mean operations.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.bins = []
        for i in range(output_size):
            start = int((i * input_size) // output_size)
            end = int(((i + 1) * input_size + output_size - 1) // output_size)
            self.bins.append((start, max(end, start + 1)))

    def forward(self, x):
        parts = [x[:, :, start:end].mean(dim=2, keepdim=True) for start, end in self.bins]
        return torch.cat(parts, dim=2)


def make_onnx_exportable(model):
    model = copy.deepcopy(model).eval()
    # b_net pools 114 subcarriers to 8 bins; c_net pools 10 packets to 4 bins.
    model.b_net[4] = FixedAdaptiveAvgPool1d(114, 8)
    model.c_net[2] = FixedAdaptiveAvgPool1d(10, 4)
    return model


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


def run(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    _, nn, _, _ = __import__("exp14_cp_cnn_aff").require_torch()
    model = CPSelectiveAFF(nn, args.temperature)().eval()
    x = torch.randn(args.batch_size, 1, RANK, 127, dtype=torch.float32)
    x_np = x.numpy()
    params = count_params(model)

    rows = []
    torch_stats = benchmark(lambda: model(x), args.warmup, args.iters)
    rows.append(
        {
            "model": "mmfi_cp_saff",
            "mode": "torch_sequential",
            "batch_size": args.batch_size,
            "params": params,
            "output_dim": OUT_DIM,
            "torch_threads": args.torch_threads,
            "intra_threads": "",
            "inter_threads": "",
            "execution_mode": "",
            "speedup_vs_torch": 1.0,
            **torch_stats,
        }
    )
    print(rows[-1])

    onnx_path = args.export_dir / "mmfi_cp_saff.onnx"
    if args.rebuild or not onnx_path.exists():
        export_onnx(make_onnx_exportable(model), x, onnx_path)

    for intra in [int(v) for v in args.intra_threads.split(",")]:
        for inter in [int(v) for v in args.inter_threads.split(",")]:
            for execution_mode in args.execution_modes.split(","):
                session = make_session(onnx_path, intra, inter, execution_mode)

                def run_onnx():
                    return session.run(None, {"x": x_np})

                stats = benchmark(run_onnx, args.warmup, args.iters)
                rows.append(
                    {
                        "model": "mmfi_cp_saff",
                        "mode": "onnx_full",
                        "batch_size": args.batch_size,
                        "params": params,
                        "output_dim": OUT_DIM,
                        "torch_threads": args.torch_threads,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--intra-threads", default="1,2,4")
    parser.add_argument("--inter-threads", default="1,2")
    parser.add_argument("--execution-modes", default="sequential,parallel")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/onnx_mmfi_saff"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mmfi_saff_runtime.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
