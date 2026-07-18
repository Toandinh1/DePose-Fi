# Current Results Summary

Last updated: 2026-07-18. Naming updated SwiftPose-Fi/S-AFF -> DePose-Fi/S-CAF.
PRA is de-scoped after the dual-CP anytime salvage failed.

Headline: a decomposition-first Wi-Fi HPE model (non-negative CP + S-CAF) that is
tiny, accurate at strict thresholds, and hardware-friendly. The paper-facing
hardware story is the measured size/FLOP/latency reduction plus the isolated CP
cost structure; proactive rank adaptation is not claimed.

## 1. Main MM-Fi Result (full protocol-3, frame-level)

| Method | PCK20 | PCK10 | MPJPE (torso-norm) | Params | FLOPs/frame |
|---|---:|---:|---:|---:|---:|
| HPE-Li (reported) | 52.07 | -- | -- | 1.66M | 2.42G |
| CP + S-CAF (ours) | **53.17** | 26.73 | 0.191 | **67.2K** | **4.90M** |

- Matches/exceeds HPE-Li at strict thresholds (+1.10 PCK20) with ~25x fewer params
  and ~493x fewer FLOPs. Full split: 224,532 train / 48,114 test.

## 1.1 Complexity + measured latency (laptop CPU, batch-1)

| Model | Params | PyTorch CPU | ONNX CPU |
|---|---:|---:|---:|
| HPE-Li (local run) | 2.06M | 15.18 ms | 3.46 ms |
| CP + S-CAF | 67.2K | 0.163 ms | 0.049 ms |

- ~93x (PyTorch) / ~71x (ONNX) faster than HPE-Li; standard ops, clean ONNX export.

## 2. Why CP (decomposition choice, matched budget, MLP64)

| Extractor | PCK20 | MPJPE | Recon. err |
|---|---:|---:|---:|
| PCA | 43.59 | 0.212 | 0.99% |
| NMF | 44.11 | 0.210 | 8.00% |
| HOSVD | 40.25 | 0.220 | 0.79% |
| CP (ours) | **46.90** | **0.201** | 11.00% |

- CP wins at matched budget despite the worst reconstruction error: judge
  decomposition by pose relevance, not reconstruction. Subcarrier factor dominates
  (dropping it: -10.12 PCK20).

## 3. Person-in-WiFi 3D (cross-dataset, tuned model)

| Method | MPJPE (mm) | Params |
|---|---:|---:|
| WiFi-Mamba (SOTA) | 76.75 | 2.14M |
| Dual amp/phase CP + S-CAF (ours) | 83.82 | 1.70M |
| original baseline | 91.77 | 48.2M |

- Below WiFi-Mamba, far above the 48.2M baseline. Multi-person disambiguation is
  the main open gap.

## 4. Hardware Contribution: Measured Deployment Cost + CP Bottleneck

The paper-facing hardware contribution should focus on measured, defensible facts:

- CP + S-CAF is ~25x smaller and ~493x fewer FLOPs than HPE-Li.
- Laptop CPU latency is ~93x faster in PyTorch and ~71x faster in ONNX.
- On-device profiling shows S-CAF is tiny; CP extraction dominates end-to-end cost.
- CP rank and CP iteration count are clean latency knobs, but only the iteration
  knob is paper-safe right now because it keeps the dedicated high-accuracy model.

## 4.1 De-scoped: Proactive Resource-Aware Rank Adaptation (PRA)

PRA is **not** paper-ready. We tried to salvage it on Person-in-WiFi 3D using the
tuned dual-CP setting:

| Model | Rank ladder | MPJPE (mm) |
|---|---|---:|
| Dedicated dual-CP S-CAF | R=24 | **83.82** |
| Anytime dual-CP, old raw-target run | R=24 | 441.04 |
| Anytime dual-CP, sandwich run | R=24 | 188.38 |
| Anytime dual-CP, standardized 80 ep | R=24 | 137.33 |
| Anytime dual-CP, standardized 160 ep | R=4/8/16/24 | 120.58 / 125.40 / 126.06 / 121.05 |

The final salvage fixed the catastrophic training mismatch but still failed the
paper bar: full rank is 121.05 mm vs the 83.82 mm dedicated model, and the rank
ladder is non-monotonic. Conclusion: do **not** claim proactive rank adaptation in
the paper. Keep it as a negative result / future work unless a stronger anytime
model recovers dedicated accuracy.

## 5. De-scoped: Branch/Thread Parallelism

Naive Python-thread branch/stream parallelism is slower than sequential at batch-1;
not claimed as a speedup and de-emphasized in the paper.

## 6. Next Experiments

1. Complete `exp36_accuracy_vs_iters`: verify whether lower CP iteration counts
   preserve the dedicated 83.82 mm PiW model's accuracy.
2. Tighten the deadline-preemptible iteration runtime text with those accuracy
   numbers; keep rank adaptation out unless it recovers.
3. ONNX INT8 / dynamic quantization vs FP32.
4. Complete `exp37_distributed_parafac` if we need a distributed-CP angle.
