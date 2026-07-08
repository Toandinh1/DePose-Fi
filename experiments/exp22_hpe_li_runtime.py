"""Benchmark HPE-Li runtime against SwiftPose-Fi deployment numbers.

This script imports the local HPE-Li ECCV 2024 implementation, measures
batch-1 PyTorch CPU latency, and attempts ONNX Runtime export/latency.

Expected local layout:
  C:/Users/toand/.openclaw/workspace/HPE-Li-ECCV2024
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install with: python -m pip install onnx onnxruntime") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPE_LI_ROOT = ROOT.parent / "HPE-Li-ECCV2024"


class HpeLiOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        pose, _ = self.model(x)
        return pose


def load_hpe_li(repo_root: Path):
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"HPE-Li repo not found: {repo_root}")
    if "torchvision" not in sys.modules:
        # HPE-Li imports torchvision.transforms.Resize in regression.py, but the
        # imported symbol is unused by the MM-Fi model. Avoid a heavy dependency
        # just for latency benchmarking.
        torchvision_stub = types.ModuleType("torchvision")
        transforms_stub = types.ModuleType("torchvision.transforms")
        transforms_stub.Resize = object
        torchvision_stub.transforms = transforms_stub
        sys.modules["torchvision"] = torchvision_stub
        sys.modules["torchvision.transforms"] = transforms_stub
    sys.path.insert(0, str(repo_root))
    from model import DSKNetTransMMFI  # noqa: WPS433

    return HpeLiOutputWrapper(DSKNetTransMMFI()).eval()


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


def export_onnx(model: nn.Module, x: torch.Tensor, path: Path):
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


def make_session(path: Path, intra_threads: int, inter_threads: int, execution_mode: str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL if execution_mode == "parallel" else ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def run(args):
    torch.set_num_threads(args.torch_threads)
    model = load_hpe_li(args.hpe_li_root)
    x = torch.randn(args.batch_size, 3, 114, 10, dtype=torch.float32)
    x_np = x.numpy()
    params = count_params(model)

    rows = []
    torch_stats = benchmark(lambda: model(x), args.warmup, args.iters)
    rows.append(
        {
            "model": "hpe_li_dsknettrans_mmfi",
            "mode": "torch_sequential",
            "batch_size": args.batch_size,
            "params": params,
            "torch_threads": args.torch_threads,
            "intra_threads": "",
            "inter_threads": "",
            "execution_mode": "",
            "speedup_vs_torch": 1.0,
            **torch_stats,
        }
    )
    print(rows[-1])

    onnx_path = args.export_dir / "hpe_li_dsknettrans_mmfi.onnx"
    if args.rebuild or not onnx_path.exists():
        export_onnx(model, x, onnx_path)

    for intra in [int(v) for v in args.intra_threads.split(",")]:
        for inter in [int(v) for v in args.inter_threads.split(",")]:
            for execution_mode in args.execution_modes.split(","):
                session = make_session(onnx_path, intra, inter, execution_mode)

                def run_onnx():
                    return session.run(None, {"x": x_np})

                stats = benchmark(run_onnx, args.warmup, args.iters)
                row = {
                    "model": "hpe_li_dsknettrans_mmfi",
                    "mode": "onnx_full",
                    "batch_size": args.batch_size,
                    "params": params,
                    "torch_threads": args.torch_threads,
                    "intra_threads": intra,
                    "inter_threads": inter,
                    "execution_mode": execution_mode,
                    "speedup_vs_torch": torch_stats["latency_mean_us"] / stats["latency_mean_us"],
                    **stats,
                }
                rows.append(row)
                print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={args.output}")
    print(f"onnx_path={onnx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpe-li-root", type=Path, default=DEFAULT_HPE_LI_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--intra-threads", default="1,2,4")
    parser.add_argument("--inter-threads", default="1,2")
    parser.add_argument("--execution-modes", default="sequential,parallel")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/onnx_hpe_li"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hpe_li_runtime.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
