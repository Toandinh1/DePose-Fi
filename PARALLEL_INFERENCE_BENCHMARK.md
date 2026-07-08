# S-AFF Branch Parallel Inference Benchmark

This note records the first Level 1/2 benchmark for the idea that S-AFF can accelerate inference through branch-level parallelism.

Script:

```bash
python experiments/exp20_saff_parallel_inference.py \
  --model-sizes small,medium,large \
  --torch-threads 1,2,4 \
  --warmup 100 \
  --iters 1000 \
  --output outputs/saff_parallel_inference_full.csv
```

## What Was Tested

The benchmark uses synthetic dual-CP Person-in-WiFi 3D-style features with rank 24.

Execution modes:

- `sequential`: amplitude stream, phase stream, and A/B/C branches run sequentially.
- `branch_parallel`: A/B/C branches run in Python threads inside each stream.
- `stream_parallel`: amplitude and phase streams run in Python threads.
- `full_parallel`: amplitude/phase streams and A/B/C branches are both threaded.

CPU thread settings:

- `torch_threads=1`
- `torch_threads=2`
- `torch_threads=4`

This approximates small edge-CPU constraints before testing on real Raspberry Pi or Jetson hardware.

## Main Result

For batch-1 inference, Python-thread branch parallelism is slower than sequential execution.

Best sequential latencies:

| Model Size | Params | Torch Threads | Mean Latency |
|---|---:|---:|---:|
| small | 131.4K | 1 | 800.49 us |
| medium | 462.2K | 1 | 902.89 us |
| large | 1.44M | 1 | 1109.69 us |

Best non-sequential result:

| Model Size | Mode | Torch Threads | Mean Latency | Speedup |
|---|---|---:|---:|---:|
| small | stream_parallel | 2 | 1748.68 us | 0.84x |
| large | stream_parallel | 1 | 1540.23 us | 0.72x |
| medium | stream_parallel | 1 | 1318.99 us | 0.68x |

All threaded modes are slower than sequential execution in this batch-1 CPU setting.

## Interpretation

S-AFF is architecturally parallelizable because its CP component branches are independent until the final gate. However, this does not automatically translate into faster inference with Python-thread execution.

The current result means:

- We should **not** claim measured speedup from branch-level parallelism yet.
- We can claim S-AFF exposes branch-level parallelism as a deployment opportunity.
- For tiny batch-1 inference, thread scheduling overhead is larger than the saved branch compute.
- Sequential S-AFF is already very fast, so parallelization must be implemented carefully to help.

## Paper-Safe Claim

Safe wording:

> S-AFF exposes branch-level parallelism because the link, subcarrier, packet, and amplitude/phase streams can be computed independently before the final lightweight gate. This structure gives a hardware scheduling opportunity that monolithic attention or state-space backbones do not expose as directly.

Unsafe wording:

> S-AFF is faster because its branches run in parallel.

We do not have evidence for that yet.

## Next Validation Steps

1. Test real Raspberry Pi 5 or Jetson hardware.
2. Try quantized ONNX Runtime inference.
3. If using parallelism, implement it below Python level, for example:
   - ONNX Runtime graph-level parallel execution,
   - C++ thread pool,
   - separate accelerator kernels,
   - hard-routing S-AFF that skips inactive branches.

## ONNX Runtime Follow-Up

Script:

```bash
python experiments/exp21_saff_onnx_parallel.py \
  --model-size large \
  --warmup 100 \
  --iters 1000 \
  --intra-threads 1,2,4 \
  --inter-threads 1,2 \
  --execution-modes sequential,parallel \
  --rebuild
```

This exports:

- a monolithic dual-CP S-AFF ONNX graph,
- an amplitude encoder ONNX graph,
- a phase encoder ONNX graph,
- a small fusion-head ONNX graph.

Best results from the laptop CPU sweep:

| Model Size | PyTorch Seq. | ONNX Full | ONNX Split Seq. | ONNX Split Stream Parallel |
|---|---:|---:|---:|---:|
| small | 969.11 us | 121.62 us | 143.02 us | 199.16 us |
| medium | 999.38 us | 178.66 us | 208.25 us | 263.32 us |
| large | 1272.71 us | 288.94 us | 372.33 us | 392.30 us |

Interpretation:

- ONNX Runtime is a strong deployment win: about 4.5x to 8x faster than PyTorch batch-1 inference.
- Splitting the architecture into amplitude/phase stream ONNX graphs works and remains fast.
- On this CPU, stream-parallel ONNX is still slower than monolithic ONNX because synchronization/session overhead dominates.
- The contribution should emphasize **deployable branch structure and scheduling flexibility**, not guaranteed CPU speedup from naive parallel execution.

Updated paper-safe claim:

> SwiftPose-Fi's S-AFF design can be exported either as a compact monolithic ONNX graph or as independent amplitude/phase component graphs plus a small fusion head. This demonstrates that the architecture is deployment-shaped: it uses standard runtime-supported operators and exposes branch-level scheduling choices, even though measured speedup from split parallel execution depends on the target runtime and hardware.
