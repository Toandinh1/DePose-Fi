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

1. Install/test ONNX Runtime CPU and compare exported sequential S-AFF latency.
2. Try a compiled backend such as TorchScript or `torch.compile` where available.
3. Test real Raspberry Pi 5 or Jetson hardware.
4. If using parallelism, implement it below Python level, for example:
   - ONNX Runtime graph-level parallel execution,
   - C++ thread pool,
   - separate accelerator kernels,
   - hard-routing S-AFF that skips inactive branches.

