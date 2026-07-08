# AAAI Paper Plan: DePose-Fi

## Core Claim

Raw Wi-Fi CSI is a complicated mixed physical signal. Directly feeding raw CSI
into a deep model is a naive abstraction: the model must learn both signal
decomposition and pose estimation at the same time.

DePose-Fi separates these two steps:

```text
raw CSI -> physical component decomposition -> tiny ML pose estimator
```

The selling point is **decomposition**. The downstream model is deliberately
simple because the decomposed representation should already expose useful
pose-relevant structure.

## One-Sentence Pitch

DePose-Fi decomposes high-dimensional Wi-Fi CSI into compact, interpretable
link-frequency-time components, enabling tiny ML regressors to achieve
competitive human pose estimation accuracy with far lower deployment cost than
deep Wi-Fi HPE models.

## Paper Structure Following AAAI Example

### Introduction

Use the AAAI example style:

1. **Motivation**
   - Wi-Fi HPE enables privacy-preserving sensing.
   - CSI is indirect and mixed.
   - HPE is harder than HAR because the output is fine-grained keypoints.

2. **Existing Limitations**
   - Existing Wi-Fi HPE models feed raw/weakly processed CSI into deep networks.
   - This forces the model to implicitly learn signal decomposition.
   - Heavy models are hard to deploy on Raspberry Pi / Jetson Nano.

3. **Challenges**
   - How to extract pose-relevant components from mixed CSI?
   - How to keep inference lightweight?
   - How to verify that components are meaningful, not just compressed numbers?

4. **Key Contributions**
   - Decomposition-first Wi-Fi HPE framework.
   - Compact link-subcarrier-packet components.
   - Tiny ML pose estimator.
   - Component quality/ablation analysis.
   - Edge-device evaluation.

## Required Result Blocks

### Result 1: Accuracy Against SOTA

Compare with Wi-Fi HPE methods:

- HPE-Li
- MetaFi / MetaFi++
- Wi-Mose
- GoPose
- Person-in-WiFi
- GraphPose-Fi if available

Metrics:

- PCK_50
- PCK_40
- PCK_30
- PCK_20
- MPJPE
- PA-MPJPE if available

Current result on accessible MM-Fi subset:

```text
CP components + ExtraTrees:
PCK_50 = 85.15
PCK_40 = 79.27
PCK_30 = 70.44
PCK_20 = 55.95
MPJPE  = 0.185
```

Need full MM-Fi result before final submission.

### Result 2: Lightweight Complexity

Compare:

- parameters
- model size
- FLOPs / operations
- inference latency
- memory usage

Against:

- HPE-Li
- MetaFi++
- our Ridge
- our MLP64
- our ExtraTrees

Expected story:

```text
Decomposition increases preprocessing slightly, but allows the pose model itself
to be extremely small and CPU-friendly.
```

Need separate timing for:

1. decomposition feature extraction,
2. ML inference,
3. end-to-end inference.

### Result 3: Edge Device Deployment

Run on:

- Raspberry Pi
- Jetson Nano

Measure:

- latency per frame,
- FPS,
- memory footprint,
- CPU utilization,
- energy if available,
- model load time.

Compare:

- HPE-Li on device,
- DePose-Fi Ridge,
- DePose-Fi MLP64,
- DePose-Fi ExtraTrees.

If HPE-Li is too slow or unavailable on Raspberry Pi, report that honestly and
compare against Jetson Nano.

### Result 4: Component Ablation

Already done:

```text
drop component 0: -7.57 PCK_20
drop component 1: -5.05 PCK_20
drop component 2: -3.89 PCK_20
drop component 3: -9.22 PCK_20
```

Need expand:

- keypoint-wise PCK drop,
- component-only performance,
- rank sweep,
- factor-mode ablation.

### Result 5: Decomposition Quality

Report:

- reconstruction error,
- factor correlation,
- component energy,
- component visualization,
- factor-mode importance.

Current:

```text
reconstruction error = 0.0863
link corr = 0.6973
subcarrier corr = 0.3832
packet corr = 0.5921
```

Problem:

Vanilla CP is predictive but not clean enough.

Required improvement:

- diversity-constrained CP,
- sparse factors,
- pose-aware decomposition.

## Figures Needed

1. Overview: raw CSI -> decomposition -> components -> tiny ML -> pose.
2. CSI tensor decomposition: X ≈ sum of link x subcarrier x packet factors.
3. Component examples: A, B, C factors.
4. Accuracy comparison with SOTA.
5. Complexity comparison.
6. Edge-device latency comparison.
7. Component ablation.
8. Factor-mode importance.

## Main Narrative

Do not sell this as "we use CP."

Sell it as:

> Wi-Fi HPE should not learn from raw mixed CSI directly. DePose-Fi first
> decomposes CSI into physically structured components, and this makes tiny ML
> pose estimation possible.

## Submission Readiness Checklist

- [ ] Full MM-Fi data downloaded.
- [ ] Full MM-Fi protocol3 results.
- [ ] SOTA comparison table.
- [ ] Complexity table.
- [ ] Raspberry Pi / Jetson Nano benchmark.
- [ ] Rank sweep.
- [ ] Keypoint-wise component ablation.
- [ ] Better component visualization.
- [ ] Related work expanded with correct citations.
- [ ] Official AAAI template compile.
