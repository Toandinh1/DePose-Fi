"""ONNX Runtime benchmark for S-AFF stream-level parallelism.

This is the next step after exp20. Python-thread PyTorch parallelism was slower
for batch-1 inference, so here we test a more deployment-realistic route:

1. Export the full dual-CP S-AFF model as one ONNX graph.
2. Export amplitude encoder, phase encoder, and fusion head as separate ONNX
   graphs.
3. Compare monolithic ONNX, split sequential ONNX, and split stream-parallel
   ONNX Runtime execution.

The split form tests the hardware-scheduling contribution: decomposition gives
independent amplitude/phase streams that can be run concurrently and synchronized
only at a small fusion head.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install with: python -m pip install onnx onnxruntime") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp20_saff_parallel_inference import JOINTS, MAX_PEOPLE, ParallelDualSaff, saff_dims  # noqa: E402


class FullWrapper(nn.Module):
    def __init__(self, model: ParallelDualSaff):
        super().__init__()
        self.model = model

    def forward(self, xa, xp):
        poses, logits, count_logits = self.model.forward_sequential(xa, xp)
        return poses, logits, count_logits


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder.forward_sequential(x)


class FusionHead(nn.Module):
    def __init__(self, model: ParallelDualSaff):
        super().__init__()
        self.cross_gate = model.cross_gate
        self.cross_fuse = model.cross_fuse
        self.query_embed = model.query_embed
        self.query_net = model.query_net
        self.pose_head = model.pose_head
        self.cls_head = model.cls_head
        self.count_head = model.count_head

    def forward(self, ha, hp):
        both = torch.cat([ha, hp], dim=1)
        stream_gate = self.cross_gate(both)
        fused = self.cross_fuse(both) + stream_gate[:, 0:1] * ha + stream_gate[:, 1:2] * hp
        q = fused.unsqueeze(1) + self.query_embed.unsqueeze(0)
        q = self.query_net(q)
        poses = self.pose_head(q).reshape((q.shape[0], q.shape[1], JOINTS, 3))
        logits = self.cls_head(q).squeeze(-1)
        count_logits = self.count_head(fused)
        return poses, logits, count_logits


def export_onnx(module: nn.Module, args, path: Path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))


def make_session(path: Path, intra_threads: int, inter_threads: int, execution_mode: str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    if execution_mode == "parallel":
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def benchmark(fn, warmup: int, iters: int):
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run(args):
    torch.set_num_threads(args.torch_threads)
    _, _, _, hidden, _, _ = saff_dims(args.model_size)
    feature_dim = args.rank * (9 + 30 + 20)
    torch.manual_seed(args.seed)
    model = ParallelDualSaff(args.rank, args.model_size, args.num_queries, args.temperature).eval()
    params = count_params(model)

    xa = torch.randn(args.batch_size, feature_dim, dtype=torch.float32)
    xp = torch.randn(args.batch_size, feature_dim, dtype=torch.float32)
    xa_np = xa.numpy()
    xp_np = xp.numpy()

    export_dir = args.export_dir / f"{args.model_size}_r{args.rank}_b{args.batch_size}"
    full_path = export_dir / "dual_saff_full.onnx"
    amp_path = export_dir / "amp_encoder.onnx"
    phase_path = export_dir / "phase_encoder.onnx"
    fusion_path = export_dir / "fusion_head.onnx"

    if args.rebuild or not all(p.exists() for p in [full_path, amp_path, phase_path, fusion_path]):
        export_onnx(FullWrapper(model), (xa, xp), full_path, ["xa", "xp"], ["poses", "logits", "count_logits"])
        export_onnx(EncoderWrapper(model.amp_encoder), (xa,), amp_path, ["x"], ["h"])
        export_onnx(EncoderWrapper(model.phase_encoder), (xp,), phase_path, ["x"], ["h"])
        h0 = torch.zeros(args.batch_size, hidden, dtype=torch.float32)
        export_onnx(FusionHead(model), (h0, h0), fusion_path, ["ha", "hp"], ["poses", "logits", "count_logits"])

    rows = []
    for intra in [int(x) for x in args.intra_threads.split(",")]:
        for inter in [int(x) for x in args.inter_threads.split(",")]:
            for execution_mode in args.execution_modes.split(","):
                full = make_session(full_path, intra, inter, execution_mode)
                amp = make_session(amp_path, intra, inter, execution_mode)
                phase = make_session(phase_path, intra, inter, execution_mode)
                fusion = make_session(fusion_path, intra, inter, execution_mode)

                def torch_seq():
                    with torch.inference_mode():
                        return model.forward_sequential(xa, xp)

                def onnx_full():
                    return full.run(None, {"xa": xa_np, "xp": xp_np})

                def onnx_split_seq():
                    ha = amp.run(None, {"x": xa_np})[0]
                    hp = phase.run(None, {"x": xp_np})[0]
                    return fusion.run(None, {"ha": ha, "hp": hp})

                with ThreadPoolExecutor(max_workers=2) as executor:
                    def onnx_split_stream_parallel():
                        ha_f = executor.submit(amp.run, None, {"x": xa_np})
                        hp_f = executor.submit(phase.run, None, {"x": xp_np})
                        ha = ha_f.result()[0]
                        hp = hp_f.result()[0]
                        return fusion.run(None, {"ha": ha, "hp": hp})

                    modes = [
                        ("torch_sequential", torch_seq),
                        ("onnx_full", onnx_full),
                        ("onnx_split_sequential", onnx_split_seq),
                        ("onnx_split_stream_parallel", onnx_split_stream_parallel),
                    ]
                    baseline = None
                    for name, fn in modes:
                        stats = benchmark(fn, args.warmup, args.iters)
                        if name == "torch_sequential":
                            baseline = stats["latency_mean_us"]
                        row = {
                            "model": "dual_cp_saff",
                            "model_size": args.model_size,
                            "rank": args.rank,
                            "batch_size": args.batch_size,
                            "params": params,
                            "mode": name,
                            "intra_threads": intra,
                            "inter_threads": inter,
                            "execution_mode": execution_mode,
                            "speedup_vs_torch": baseline / stats["latency_mean_us"],
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
    print(f"export_dir={export_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--model-size", default="large", choices=["small", "medium", "large"])
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--intra-threads", default="1,2,4")
    parser.add_argument("--inter-threads", default="1,2")
    parser.add_argument("--execution-modes", default="sequential,parallel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/onnx_saff"))
    parser.add_argument("--output", type=Path, default=Path("outputs/saff_onnx_parallel.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
