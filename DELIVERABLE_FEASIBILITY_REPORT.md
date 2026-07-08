# Feasibility Deliverable: CSI Decomposition for Tiny Wi-Fi HPE

## Goal

Prove that the idea is not just theoretical:

> Decompose Wi-Fi CSI into compact components, then use a super-lightweight ML
> model for human pose estimation.

The current evidence uses the HPE-Li-style MM-Fi frame protocol:

```text
input per sample: one CSI frame, CSIamp shape 3 x 114 x 10
target: 17 human keypoints
metric: HPE-Li PCK_50/PCK_40/PCK_30/PCK_20/PCK_10
```

Important limitation:

```text
available local frames: 178,497
expected full MM-Fi frames: 320,760
```

The local Kaggle extraction is incomplete, so results are fair by protocol but
not yet full-MM-Fi final results.

## Evidence 1: Lightweight ML Works On Decomposed Components

Using rank-4 CP components:

```math
X(l,s,p) \approx \sum_{r=1}^{4} A(l,r)B(s,r)C(p,r)
```

The CP feature vector is:

```text
A: 3 x 4
B: 114 x 4
C: 10 x 4
total: 508 values
```

Raw frame size:

```text
3 x 114 x 10 = 3420 values
```

Compression:

```text
3420 / 508 = 6.73x
```

### Main Result

| Method | PCK_50 | PCK_40 | PCK_30 | PCK_20 | PCK_10 | MPJPE |
|---|---:|---:|---:|---:|---:|---:|
| Mean pose | 82.82 | 74.75 | 63.02 | 45.81 | 17.36 | 0.208 |
| Raw stats + Ridge | 83.55 | 76.94 | 67.05 | 49.61 | 20.08 | 0.200 |
| Raw stats + MLP64 | 84.21 | 78.05 | 68.57 | 52.08 | 22.72 | 0.197 |
| Raw stats + ExtraTrees | 84.74 | 78.37 | 69.18 | 53.81 | 26.07 | 0.190 |
| CP components + Ridge | 83.38 | 76.66 | 66.52 | 49.09 | 19.64 | 0.201 |
| CP components + MLP64 | 83.90 | 77.54 | 67.84 | 51.59 | 22.89 | 0.198 |
| **CP components + ExtraTrees** | **85.15** | **79.27** | **70.44** | **55.95** | **28.74** | **0.185** |
| HPE-Li reported | 85.12 | 78.18 | 68.22 | 52.07 | - | - |

Interpretation:

- Decomposed components + ExtraTrees are competitive with HPE-Li reported
  performance.
- This supports the tiny-ML thesis.
- However, raw stats + ExtraTrees is also strong, so the paper must prove that
  decomposition adds more than generic feature compression.

## Evidence 2: Components Are Pose-Relevant

We trained ExtraTrees on all CP components, then removed one component at test
time.

| Ablation | PCK_20 | Drop vs All |
|---|---:|---:|
| All components | 55.95 | 0.00 |
| Drop component 0 | 48.38 | -7.57 |
| Drop component 1 | 50.90 | -5.05 |
| Drop component 2 | 52.06 | -3.89 |
| Drop component 3 | 46.73 | -9.22 |

Interpretation:

- Every component matters.
- Component 3 is most pose-relevant.
- Component 0 is also important.
- This is strong evidence that the CP components are not decorative.

## Evidence 3: Different Factor Modes Have Different Roles

We removed entire factor modes from the CP feature vector.

| Dropped Factor Mode | PCK_20 | Drop vs All |
|---|---:|---:|
| None | 55.95 | 0.00 |
| Link factor `A` | 55.28 | -0.67 |
| Subcarrier factor `B` | 44.17 | -11.78 |
| Packet factor `C` | 55.70 | -0.25 |

Feature importance:

| Group | Importance |
|---|---:|
| Link factor `A` | 0.024 |
| Subcarrier factor `B` | 0.946 |
| Packet factor `C` | 0.030 |

Interpretation:

- The current frame-level decomposition is dominated by the subcarrier factor.
- This makes sense because a single MM-Fi frame contains only 10 packets, so
  the short-time packet mode is weak.
- For physical motion components, we likely need a temporal/Doppler-window
  version, but that requires equal-window comparison with HPE-Li.

## Evidence 4: Reconstruction Is Good, But Separation Is Not Clean Enough

On 3,000 sampled frames:

```text
mean reconstruction error = 0.0863
link factor correlation = 0.6973
subcarrier factor correlation = 0.3832
packet factor correlation = 0.5921
```

Interpretation:

- CP preserves CSI structure well.
- But factors are still correlated, especially link and packet factors.
- Vanilla CP is not enough for a final paper contribution.

## Go / No-Go

Decision: **GO, but pivot to constrained pose-aware decomposition.**

The idea is doable because:

1. compact CP components support strong HPE with tiny ML;
2. component ablation causes large PCK drops;
3. subcarrier/factor analysis gives physical insight;
4. results are competitive with HPE-Li on available protocol3 frame samples.

But the paper is not ready if we only use vanilla CP.

## Required Next Experiments

### Experiment A: Complete Full MM-Fi

We need all 320,760 frame samples before making final HPE-Li comparison.

### Experiment B: Rank Sweep

Run:

```text
R = 2, 4, 6, 8, 12
```

Report:

- PCK/MPJPE,
- reconstruction error,
- feature dimension,
- inference time,
- component correlations.

### Experiment C: Pose-Aware / Constrained Decomposition

Vanilla CP should be replaced or augmented with constraints:

```math
\mathcal{L}
=
\mathcal{L}_{rec}
+ \lambda_{pose}\mathcal{L}_{pose}
+ \lambda_{div}\mathcal{L}_{div}
+ \lambda_s\mathcal{L}_{sparse}
+ \lambda_t\mathcal{L}_{smooth}
```

The main objective is to reduce factor correlation and increase pose relevance.

### Experiment D: Component Visualization

For each component:

- plot link factor `A`,
- plot subcarrier factor `B`,
- plot packet/time factor `C`,
- show component energy by action,
- show keypoint-wise PCK drop when removed.

### Experiment E: Temporal/Doppler Equal-Window Study

Frame-level comparison is fair to HPE-Li, but weak for motion decomposition.
For deeper physical decomposition, we need:

```text
same temporal window for HPE-Li, raw LDT, NMF, CP, and constrained DePose-Fi
```

Then we can study true Doppler motion components.

## Paper Claim After This Deliverable

Safe claim:

> Decomposed CSI components provide a compact and pose-relevant representation
> that enables classical ML models to approach neural Wi-Fi HPE accuracy under
> frame-level MM-Fi evaluation.

Not yet safe:

> The components are clean physical body-part motion sources.

To make that safe, we need constrained/pose-aware decomposition and
visualization.

## Output Files

- `outputs/component_ml_bench_available.csv`
- `outputs/component_pose_relevance_ablation.csv`
- `outputs/component_pose_relevance_importance.csv`
- `outputs/component_quality_probe.csv`
