# Decompose WiFi Sensing HPE - Publication Plan

## Working Title

**DePose-Fi: Tensor-Factorized CSI Representation for Wi-Fi Human Pose Estimation**

## Core Claim

Tensor factorization provides a source-aware CSI representation by decomposing mixed wireless observations into latent motion components. These components can improve CSI-based human pose heatmap estimation compared with raw dynamic CSI, raw link-Doppler-time tensors, and matrix NMF features.

Important: do **not** claim perfect person-level or body-part-level separation. The defensible claim is that tensor factors expose useful latent motion structure for downstream HPE.

## Why This May Be Publishable

Most Wi-Fi HPE work maps CSI directly to pose. The proposed paper inserts an interpretable decomposition layer before pose estimation:

1. Remove static CSI.
2. Convert CSI into a motion-aware link-Doppler-time tensor.
3. Apply non-negative CP tensor factorization.
4. Reconstruct component-wise CSI motion tensors.
5. Feed factorized components into the same heatmap estimator used by baselines.

The novelty is strongest if experiments show improved:

- cross-person or multi-person robustness,
- high-mobility keypoint accuracy, especially wrists/elbows/ankles,
- keypoint confusion reduction in two-person scenes,
- interpretability of factors through link, Doppler, and temporal activations.

## Mathematical Formulation

### CSI Mixture

For subcarrier/link/time CSI:

```math
H(s,l,t)=H_{static}(s,l)+\sum_{k=1}^{K}H_k(s,l,t)+N(s,l,t).
```

The method does not try to exactly recover each `H_k`. Instead, it extracts latent motion factors useful for pose estimation.

### Static Removal

```math
\bar{H}(l,f)=\frac{1}{T}\sum_{t=1}^{T}H(l,f,t),
```

```math
\Delta H(l,f,t)=H(l,f,t)-\bar{H}(l,f).
```

### Link-Doppler-Time Tensor

Apply STFT over time and aggregate over frequency:

```math
X(l,d,\tau)=\sum_{f=1}^{N_f}\left|\mathrm{STFT}_t(\Delta H(l,f,t))\right|^2.
```

This gives:

```math
X\in\mathbb{R}_{+}^{L\times D\times T'}.
```

### Non-Negative CP Factorization

```math
X(l,d,\tau)\approx\hat{X}(l,d,\tau)=\sum_{r=1}^{R}A(l,r)B(d,r)C(\tau,r),
```

where:

- `A(:,r)` is the link/spatial signature,
- `B(:,r)` is the Doppler signature,
- `C(:,r)` is the temporal activation.

### Constrained Objective

```math
\mathcal{L}_{rec}=\|X-\hat{X}\|_F^2,
```

```math
\mathcal{L}_{smooth}=\sum_{r=1}^{R}\sum_{\tau=2}^{T'}|C(\tau,r)-C(\tau-1,r)|,
```

```math
\mathcal{L}_{sparse}=\|C\|_1,
```

```math
\mathcal{L}_{fact}
=
\mathcal{L}_{rec}
+\lambda_1\mathcal{L}_{smooth}
+\lambda_2\mathcal{L}_{sparse}.
```

### Pose Estimation

Component-wise tensors:

```math
X_r(l,d,\tau)=A(l,r)B(d,r)C(\tau,r).
```

Stack:

```math
Z=[X_1,X_2,\ldots,X_R]\in\mathbb{R}_{+}^{R\times L\times D\times T'}.
```

Heatmap prediction:

```math
\hat{Y}=F_\theta(Z).
```

Pose loss:

```math
\mathcal{L}_{pose}=\sum_{j=1}^{J}\|\hat{Y}_j-Y_j\|_2^2.
```

For the first paper version, use offline factorization and train only the pose network. A stronger later version can make factorization differentiable:

```math
\mathcal{L}_{total}
=
\mathcal{L}_{pose}
+\alpha\mathcal{L}_{rec}
+\beta\mathcal{L}_{smooth}
+\gamma\mathcal{L}_{sparse}.
```

## Baselines

Use the same pose network for all input representations:

1. Raw dynamic CSI power:

```math
X_{raw}(l,f,t)=|\Delta H(l,f,t)|^2.
```

2. Raw link-Doppler-time tensor:

```math
X(l,d,\tau).
```

3. Matrix NMF after unfolding the tensor:

```math
X_{matrix}\approx WH.
```

4. Proposed non-negative CP tensor components:

```math
X\approx\sum_{r=1}^{R}A(:,r)\circ B(:,r)\circ C(:,r).
```

## Experiments Needed

### E1: Synthetic Factor Recovery

Goal: verify the decomposition assumption under controlled data.

Current synthetic probe result:

- tensor shape: `(8, 32, 80)`
- relative reconstruction error: `0.2212`
- mean matched link-factor correlation: `0.869`
- mean matched Doppler-factor correlation: `0.871`
- mean matched temporal-factor correlation: `0.857`

Files:

- `synthetic_cp_probe.py`
- `synthetic_cp_probe.png`

Interpretation: CP can recover structured link/Doppler/time factors in a clean synthetic setting, but it is not perfect. This supports the latent-motion-component claim, not a perfect source-separation claim.

### E2: Single-Person HPE

Goal: test whether factorized input improves pose heatmap prediction.

Report:

- PCK@5, PCK@10, PCK@20,
- MPJPE if coordinates are decoded,
- keypoint-wise PCK for wrists, elbows, knees, ankles.

### E3: Component Number Sensitivity

Test:

```text
R = 2, 4, 6, 8, 12
```

Report both:

- factor reconstruction error,
- downstream HPE accuracy.

Do not choose `R` only by reconstruction error. The paper cares about pose.

### E4: Constraint Ablation

Compare:

- unconstrained CP,
- CP + temporal smoothness,
- CP + temporal sparsity,
- CP + both constraints.

### E5: Matrix NMF vs Tensor CP

This is crucial. The paper's core is tensor structure, so it must beat matrix flattening.

### E6: Two-Person / Keypoint Confusion

Only after E2 works.

Measure:

- PCK drop from one person to two people,
- keypoint association errors,
- qualitative confusion examples.

## Risks

1. **CP factors may not align with people or limbs.**
   - Mitigation: claim latent motion components, not semantic body-part separation.

2. **Factorization may remove fine-grained pose cues.**
   - Mitigation: sweep `R` and compare downstream HPE, not only reconstruction.

3. **A weak pose network may hide representation benefits.**
   - Mitigation: use the exact same network across all baselines.

4. **Offline factorization may be slow.**
   - Mitigation: first establish accuracy; later add runtime analysis or differentiable/faster factorization.

## Publication Strategy

The strongest story is:

> Wi-Fi HPE suffers because CSI is a mixed multipath observation. Instead of forcing a neural network to learn pose directly from this mixture, we decompose motion-aware CSI into latent link-Doppler-time factors. These factors provide a more interpretable and useful input representation for pose heatmap estimation, especially for high-mobility keypoints and multi-person confusion.

Target paper structure:

1. Introduction
2. Related Work
3. CSI Motion Tensor Representation
4. Tensor-Factorized Motion Decomposition
5. Pose Heatmap Estimation
6. Experiments
7. Discussion and Limitations
8. Conclusion

## Immediate Next Actions

1. Locate or create data-loading scripts for MM-Fi/WiPose CSI and labels.
2. Implement static removal and STFT tensor construction.
3. Run E1 synthetic probe repeatedly across noise/rank settings.
4. Build a simple shared CNN heatmap estimator.
5. Compare raw Doppler tensor vs CP-factorized components on single-person samples.
