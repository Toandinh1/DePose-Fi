# WiDeFus Comparison

Paper:

```text
WiDeFus: A Wi-Fi-Based Lightweight Human Activity Recognition via CSI
Component Decomposition and Adaptive Feature Fusion
IEEE TCCN, 2026
```

Local files:

- `WiDeFus.pdf`
- `WiDeFus.txt`

## What WiDeFus Does

WiDeFus targets Wi-Fi CSI-based human activity recognition (HAR). Its central
claim is that raw CSI contains human-path components, static interference paths,
dynamic interference paths, and noise. It argues that deep HAR models become
heavy because they learn from mixed CSI.

The pipeline is:

1. Use CSI phase difference between adjacent time points.
2. Model phase difference as:

```math
\Delta \theta(t) = \Delta \theta_h(t) + \Delta \theta_d(t) + \Delta \theta_n(t)
```

where:

- `h`: human path
- `d`: dynamic interference path
- `n`: noise

3. Use Hermite-Gaussian basis functions and sparse optimization:

```math
\min_{C_h,C_d}
\|\Theta - H(C_h+C_d)\|_F^2
+ \lambda_h \|C_h\|_2
+ \lambda_d \|C_d\|_1
```

4. Reconstruct the human-path phase component.
5. Feed the purified component into triple-feature adaptive fusion:
   - frequency-domain spectral attention,
   - time-domain dilated causal convolution,
   - spatial/domain adversarial calibration.
6. Classify activities using a dendrite network.

## Why It Is Close To Our Idea

The overlap is real:

- Both start from the same criticism: raw CSI is mixed and contains nuisance
  components.
- Both use decomposition before prediction.
- Both argue that cleaner CSI features enable lightweight models.
- Both emphasize physical motivation and generalization.

If our paper only says "decompose CSI and use a lightweight model," WiDeFus
would weaken the novelty badly.

## Key Differences We Can Use

### 1. Task Difference

WiDeFus is HAR classification:

```text
CSI -> activity label
```

Our target is human pose estimation:

```text
CSI -> 17 human keypoints
```

HPE is finer-grained and requires continuous keypoint regression, not only
activity-level discrimination.

### 2. Decomposition Target Difference

WiDeFus extracts one purified human-path phase component:

```math
\Delta \theta_h(t)
```

Our target should be multi-component pose-relevant decomposition:

```math
X(l,s,p) \approx \sum_{r=1}^{R} A(l,r)B(s,r)C(p,r)
```

or, for temporal windows:

```math
X(l,d,\tau) \approx \sum_{r=1}^{R} A(l,r)B(d,r)C(\tau,r)
```

The components are not only "human vs interference"; they are latent
pose-relevant wireless motion components with link, frequency/Doppler, and
time factors.

### 3. Structure Difference

WiDeFus uses a Hermite-Gaussian temporal basis over phase-difference signals.
It does not preserve the full multi-way CSI tensor structure as link by
subcarrier/Doppler by time.

Our method should explicitly preserve tensor modes:

- link factor: spatial visibility,
- subcarrier/frequency or Doppler factor: multipath/motion signature,
- packet/time factor: temporal activation.

### 4. Output/Backbone Difference

WiDeFus still uses a tailored neural fusion network and dendrite classifier.
Our stronger angle is:

```text
physically interpretable components + traditional ML regressor
```

This is attractive for tiny-device deployment.

### 5. Evaluation Difference

WiDeFus evaluates activity accuracy and cross-domain HAR. Our paper should
evaluate:

- PCK_50/PCK_40/PCK_30/PCK_20,
- MPJPE and PA-MPJPE,
- keypoint-wise accuracy,
- component ablation,
- component reconstruction,
- component diversity,
- component pose relevance,
- inference latency and model size.

This gives us a stronger decomposition-quality story.

## Novelty Risk

Risk level: **medium-high** if the method remains vanilla CP.

WiDeFus already claims:

> CSI component decomposition + lightweight model improves Wi-Fi sensing.

Therefore, our paper must not be framed as a generic decomposition idea.

## Required Pivot

We should frame our paper as:

> Pose-aware physical CSI component decomposition for lightweight Wi-Fi HPE.

The key words are:

- **pose-aware**: components are evaluated and eventually optimized for keypoint
  estimation, not only signal purification;
- **multi-component**: link/frequency-time or link-Doppler-time factors, not a
  single human-path phase reconstruction;
- **HPE-specific**: continuous keypoint regression and keypoint-wise analysis;
- **tiny ML deployment**: simple regressors after decomposition.

## How To Cite/Position WiDeFus

In Related Work:

> Recent Wi-Fi HAR methods have begun to purify CSI before classification.
> WiDeFus reconstructs a human-path phase component using Hermite-Gaussian
> sparse decomposition and then applies adaptive feature fusion for lightweight
> HAR. Unlike WiDeFus, our objective is fine-grained human pose estimation. We
> decompose CSI into multiple pose-relevant link-frequency/time or
> link-Doppler/time components and evaluate whether these components support
> lightweight keypoint regression.

## What We Must Prove Experimentally

1. Decomposition helps HPE, not just HAR.
2. Same ML backbone performs better on decomposed components than raw features.
3. Components are compact and interpretable.
4. Component ablation causes measurable PCK drops.
5. The method remains competitive with HPE-Li under the full MM-Fi protocol3
   frame-level comparison.
6. Optional but strong: constrained/pose-aware decomposition improves over
   vanilla CP.

## Bottom Line

WiDeFus is close enough that we must cite it and sharpen the novelty. But it
does not kill the idea. The defensible contribution is not "CSI decomposition
for lightweight Wi-Fi sensing"; it is:

> multi-component, pose-aware CSI decomposition for tiny Wi-Fi human pose
> estimation.
