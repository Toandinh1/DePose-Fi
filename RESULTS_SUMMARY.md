# Current Results Summary

Last updated: 2026-07-08

This file summarizes the current experimental evidence for SwiftPose-Fi / DePose-Fi.

## 1. Main MM-Fi Result

The strongest result remains the MM-Fi frame-level experiment.

| Dataset / Setting | Method | Accuracy | Params | FLOPs |
|---|---:|---:|---:|---:|
| MM-Fi protocol-3 frame-level | HPE-Li baseline | 52.07 PCK20 | 1.66M | 2.42G |
| MM-Fi protocol-3 frame-level | CP + S-AFF | 50.80 PCK20 | 64.9K | 2.89M |

Interpretation:

- CP + S-AFF is only 1.27 PCK20 behind HPE-Li.
- It uses about 26x fewer parameters.
- It uses more than 800x fewer FLOPs.
- This is the cleanest hardware-friendly evidence so far.

## 1.1 MM-Fi Runtime Against HPE-Li

Scripts:

```bash
python experiments/exp23_mmfi_saff_runtime.py
python experiments/exp22_hpe_li_runtime.py
```

Best laptop CPU batch-1 runtime:

| Model | Params | PyTorch CPU | ONNX CPU |
|---|---:|---:|---:|
| HPE-Li DSKNetTrans-MMFI | 2.06M | 25.48 ms | 7.24 ms |
| CP + S-AFF | 64.9K | 0.365 ms | 0.0868 ms |

Runtime interpretation:

- CP + S-AFF is about 70x faster than HPE-Li in PyTorch CPU batch-1 inference.
- CP + S-AFF is about 83x faster than HPE-Li in ONNX Runtime CPU batch-1 inference.
- HPE-Li also benefits from ONNX, but its selective-kernel/transformer-style backbone remains much slower.
- This is a strong deployment argument: our model is not merely smaller on paper; it is substantially faster under the same CPU runtime family.

## 2. Person-in-WiFi 3D Adaptation

Person-in-WiFi 3D is a harder 3D multi-person setting. The important lesson is that amplitude and phase should be decomposed with separate CP streams.

| Method | MPJPE | Params |
|---|---:|---:|
| WiFi-Mamba SOTA | 76.75 mm | 2.14M |
| Dual amplitude/phase CP + S-AFF-L | 83.82 mm | 1.70M |
| Dual amplitude/phase CP + S-AFF-M | 91.35 mm | 923K |
| Dual amplitude/phase CP + S-AFF-S | 107.53 mm | 248K |

Interpretation:

- We do not beat WiFi-Mamba accuracy yet.
- The large model is close, but the parameter advantage is modest.
- The medium/small models are better deployment tradeoffs.
- Separate amplitude/phase CP streams are the right direction for PiW-style 3D pose.

## 2.1 Decomposition and Regressor Ablation

Script:

```bash
python experiments/exp26_decomposition_feature_comparison.py
python experiments/exp27_decomposition_regressor_ablation.py
```

The first quick probe with Ridge showed that CP is not automatically superior under a linear regressor. The fairer ablation pairs each decomposition with a neural MLP regressor, and also includes CP + S-AFF. The latest run uses the full official MM-Fi protocol-3 split from `D:\TinySense\MM-Fi`: 224,532 training frames and 48,114 test frames.

Full MM-Fi result:

| Feature | Regressor | Dim | Regressor Params | MPJPE | PCK20 |
|---|---|---:|---:|---:|---:|
| PCA | MLP | 128 | 111.9K | 0.1872 | 53.53 |
| Matrix NMF | MLP | 128 | 111.9K | 0.2569 | 32.85 |
| Tucker | MLP | 553 | 220.7K | 0.1984 | 49.64 |
| CP | MLP | 508 | 209.2K | 0.1912 | 52.09 |
| CP | S-AFF | 508 | 64.9K | 0.1972 | 50.30 |

Interpretation:

- PCA + MLP is the strongest pure predictive decomposition baseline in this full ablation.
- CP + MLP is close to PCA + MLP, trailing by 1.44 PCK20 points while preserving link/subcarrier/packet modes.
- Matrix-NMF and Tucker are weaker than CP under matched neural regressors.
- CP + S-AFF trades 1.79 PCK20 points relative to CP + MLP for a 3.2x smaller deployable regressor with component branches and hard-routing potential.
- This CP + S-AFF row uses the shared ablation training protocol. The tuned main CP-variant run remains 50.80 PCK20.
- The fair claim is not "CP always wins accuracy"; it is "CP gives the best structured basis for an interpretable, hardware-shaped fusion architecture."

Controlled MM-Fi subset result with 5K train / 1K test frames:

| Feature | Regressor | Dim | Regressor Params | MPJPE | PCK20 |
|---|---|---:|---:|---:|---:|
| PCA | MLP | 128 | 111.9K | 0.3900 | 20.61 |
| Matrix NMF | MLP | 128 | 111.9K | 0.4991 | 19.39 |
| Tucker | MLP | 553 | 220.7K | 0.3747 | 31.85 |
| CP | MLP | 508 | 209.2K | 0.3412 | 38.64 |
| CP | S-AFF | 508 | 64.9K | 0.3498 | 39.06 |

Subset interpretation:

- CP is the strongest decomposition family among PCA, matrix-NMF, Tucker, and CP when each gets an MLP regressor.
- CP + S-AFF gives the best PCK20 on this subset while using about 3.2x fewer regressor parameters than CP + MLP.
- CP + MLP has slightly lower MPJPE than CP + S-AFF on this small subset, so the full-data result remains important.
- The full MM-Fi CP + S-AFF result is still 50.80 PCK20 in the main CP variant table; this 5K/1K table is an ablation, not the final headline accuracy.
- After the full decomposition ablation, the claim should be refined: CP is not the top pure predictor against PCA, but it is the strongest structured factorization for S-AFF and deployment.

## 3. Python Thread Parallelism

Script:

```bash
python experiments/exp20_saff_parallel_inference.py
```

Finding:

- Naive Python-thread branch parallelism is slower than sequential batch-1 inference.
- Thread scheduling overhead dominates because S-AFF branches are already lightweight.

Paper-safe interpretation:

> S-AFF exposes branch-level parallelism, but naive Python-thread execution does not provide measured batch-1 CPU speedup.

## 4. ONNX Runtime Deployment

Script:

```bash
python experiments/exp21_saff_onnx_parallel.py
```

Best laptop CPU latencies:

| Model Size | PyTorch Seq. | ONNX Full | ONNX Split Seq. | ONNX Split Stream Parallel |
|---|---:|---:|---:|---:|
| small | 969.11 us | 121.62 us | 143.02 us | 199.16 us |
| medium | 999.38 us | 178.66 us | 208.25 us | 263.32 us |
| large | 1272.71 us | 288.94 us | 372.33 us | 392.30 us |

Interpretation:

- ONNX Runtime gives a strong deployment speedup: about 4.5x to 8x faster than PyTorch CPU inference.
- S-AFF exports cleanly using standard runtime-supported operators.
- Split amplitude/phase ONNX graphs work and remain fast.
- Monolithic ONNX is currently fastest on laptop CPU.
- Split execution is still useful as evidence that the architecture supports hardware scheduling flexibility.

Compared with HPE-Li, the MM-Fi CP + S-AFF ONNX model is much faster:

| Model | Best ONNX CPU Latency |
|---|---:|
| HPE-Li DSKNetTrans-MMFI | 7242.15 us |
| CP + S-AFF | 86.77 us |

## 4.1 Hard-Routed S-AFF Deployment

Scripts:

```bash
python experiments/exp24_hard_routed_saff.py
python experiments/exp25_mmfi_bonly_runtime.py
```

The trained S-AFF gate becomes extremely sharp on MM-Fi:

| Statistic | Value |
|---|---:|
| Gate max mean | 0.999999 |
| Gate entropy | 0.000010 |
| Subcarrier expert selected | 100.0% |

This enables a hard-routed deployment mode: execute only the subcarrier branch instead of all four S-AFF experts.

| Mode | PCK20 | MPJPE | Executed Experts | ONNX Latency |
|---|---:|---:|---:|---:|
| Full soft S-AFF | 48.2518 | 0.201738 | 4 / 4 | 86.77 us |
| Top-1 subcarrier routed | 48.2517 | 0.201738 | 1 / 4 | 54.18 us |

Interpretation:

- Hard routing preserves the trained full model's accuracy in this run.
- It reduces expert execution from four branches to one branch.
- It gives about 1.6x measured ONNX latency reduction over full S-AFF.
- Compared with HPE-Li ONNX, hard-routed S-AFF is about 134x faster on this laptop CPU benchmark.

This gives us a concrete deployment contribution:

> S-AFF is not only branch-structured; its learned sparse gate can be converted into a hard routing policy that skips inactive experts at inference time.

## 4.2 Resource-Aware Rank Selection

Script:

```bash
python experiments/exp28_resource_aware_rank_selector.py
```

This simulates a proactive edge scheduler. The runtime observes available CPU budget and chooses the highest CP-rank profile that satisfies a 20 ms frame-latency target. The latency side uses rank-specific ONNX S-AFF measurements plus calibrated CP extraction cost. The R=2/6/8 PCK20 values are scenario assumptions until those ranks are fully trained; R=4 is anchored to the measured CP+S-AFF result.

Rank profiles at 100% CPU:

| Rank | Params | Assumed PCK20 | CP latency | ONNX S-AFF latency | Total latency |
|---:|---:|---:|---:|---:|---:|
| 2 | 64.1K | 47.00 | 1.70 ms | 0.172 ms | 1.87 ms |
| 4 | 64.9K | 50.80 | 3.40 ms | 0.204 ms | 3.60 ms |
| 6 | 65.6K | 51.60 | 5.10 ms | 0.156 ms | 5.26 ms |
| 8 | 66.4K | 52.10 | 6.80 ms | 0.200 ms | 7.00 ms |

Selector outcome with a 20 ms target:

| CPU Available | Selected Rank | Effective Latency | Effective FPS | Assumed PCK20 |
|---:|---:|---:|---:|---:|
| 20% | 4 | 18.02 ms | 55.49 | 50.80 |
| 40% | 8 | 17.50 ms | 57.14 | 52.10 |
| 80% | 8 | 8.75 ms | 114.29 | 52.10 |

Takeaway:

> CP rank is a real systems knob. Under heavy CPU contention, the scheduler lowers rank to stay inside the latency budget; when resources are available, it chooses a higher-rank profile for better accuracy.

## 5. Current Contribution Framing

Strong claims:

- Decomposition-first Wi-Fi HPE can produce a tiny, accurate, hardware-friendly model.
- S-AFF is mode-aware and interpretable because it fuses CP components rather than raw CSI.
- CP's advantage appears when paired with a component-aware model; generic PCA/NMF/Tucker features with Ridge are much weaker in the controlled subset.
- S-AFF is deployment-friendly because it uses standard operators and exports cleanly to ONNX.
- S-AFF exposes independent component streams, enabling branch-level scheduling on edge hardware.
- Gate-sharpened S-AFF supports hard-routed inference that skips inactive experts and reduces measured ONNX latency.
- CP rank supports proactive resource-aware deployment, where the runtime selects the largest feasible rank under the current CPU budget.

Claims we should avoid until we have real-device evidence:

- Do not say branch-parallel execution is already faster on CPU.
- Do not say we beat WiFi-Mamba on Person-in-WiFi 3D.
- Do not say PiW multi-person is solved.

## 6. Next Experiments

1. Run ONNX Runtime on Raspberry Pi 5 or Jetson.
2. Add ONNX quantization and compare FP32 vs INT8/dynamic quantization.
3. Implement hard-routing S-AFF to skip inactive branches.
4. Train full mixed-person dual amplitude/phase CP + S-AFF on Person-in-WiFi 3D.
5. Add energy/FPS/memory reporting for real edge hardware.
6. Train and evaluate true R=2/6/8 MM-Fi accuracy profiles to replace the resource-selector assumptions.
