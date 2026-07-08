# Deep Decomposition Direction

## Core Thesis

Wi-Fi CSI for HPE is a mixture of propagation paths, static environment,
human-induced motion, and noise. If this mixed CSI can be decomposed into
physically meaningful components, pose estimation can be performed by a
super-lightweight ML backbone instead of a heavy neural network.

The paper should be centered on decomposition, not on model architecture.

## Physical CSI Model

For one CSI frame, MM-Fi provides:

```text
X in R_+^{L x S x P}
```

where:

- `L`: antenna/link dimension, e.g., 3 links
- `S`: subcarrier dimension, e.g., 114 subcarriers
- `P`: short packet/time dimension, e.g., 10 packets per frame

We model the normalized CSI amplitude as:

```math
X(l,s,p) \approx \sum_{r=1}^{R} a_r(l)b_r(s)c_r(p) + E(l,s,p)
```

Each component has a physical interpretation:

- `a_r(l)`: link visibility or spatial sensitivity
- `b_r(s)`: frequency/subcarrier selectivity, related to multipath structure
- `c_r(p)`: short-time activation inside the frame
- component energy: strength of the corresponding latent propagation/motion mode

For temporal-window experiments, the tensor becomes:

```math
X(l,d,\tau) \approx \sum_{r=1}^{R} a_r(l)b_r(d)c_r(\tau)
```

where:

- `a_r(l)`: link visibility
- `b_r(d)`: Doppler/motion-speed signature
- `c_r(\tau)`: temporal activation

The frame-level version is fair to HPE-Li; the Doppler-window version is more
physically meaningful for motion decomposition.

## Why Simple ML Can Work

The decomposed representation reduces the burden on the pose estimator. Instead
of learning from raw CSI samples, the ML model receives compact component
features:

```math
z = [A(:), B(:), C(:), e]
```

where `e` contains component energies and optional statistics. If the
decomposition is meaningful, a simple model such as Ridge, PLS, ExtraTrees, or a
tiny MLP can map these factors to pose keypoints.

The current result supports this:

```text
CP components + ExtraTrees:
PCK_50 = 85.148
PCK_40 = 79.273
PCK_30 = 70.442
PCK_20 = 55.947
```

This is competitive with the HPE-Li reported result while using a classical ML
regressor.

## How To Evaluate Decomposition Quality

We need evaluate both **predictive quality** and **physical/component quality**.

### 1. Predictive Quality

Use the same lightweight backbone on raw and decomposed features:

- raw stats + Ridge vs CP components + Ridge
- raw stats + MLP vs CP components + MLP
- raw stats + ExtraTrees vs CP components + ExtraTrees
- HPE-Li baseline

If the same backbone performs better on decomposed features, the decomposition
is useful.

### 2. Compactness

Report:

```text
raw feature dimension / decomposed feature dimension
model parameters
inference latency
memory footprint
```

Current frame-level CP rank-4 compression:

```text
raw CSI frame: 3 x 114 x 10 = 3420 values
CP factors: 3x4 + 114x4 + 10x4 = 508 values
compression ratio: 6.73x
```

### 3. Reconstruction Quality

Report normalized reconstruction error:

```math
\frac{\|X-\hat{X}\|_F}{\|X\|_F}
```

Current sampled result:

```text
mean CP reconstruction error = 0.0863
```

This means the components preserve most CSI energy.

### 4. Component Separation

Report factor diversity:

```math
\mathrm{corr}(A), \mathrm{corr}(B), \mathrm{corr}(C)
```

Low correlation means cleaner separation. Current result:

```text
link factor correlation = 0.6973
subcarrier factor correlation = 0.3832
packet factor correlation = 0.5921
```

This is the weak point. The current decomposition reconstructs well but is not
cleanly separated enough.

### 5. Physical Interpretability

Visualize:

- link factors `A`: which antenna links are sensitive
- subcarrier factors `B`: frequency-selective multipath patterns
- packet/time factors `C`: short-time motion activations
- component energy across actions and keypoints

For Doppler-window experiments, visualize:

- Doppler peaks in `B`
- temporal activations in `C`
- alignment between high-energy components and motion-heavy actions

### 6. Pose-Relevance

Measure how each component contributes to pose:

- train with all components
- remove one component at a time
- keep only one component at a time
- report PCK drop per component

This tells us whether components are actually useful for HPE.

## How To Make Components Physically Cleaner

Plain CP is not enough. We need constrained decomposition:

### Sparse Link Factor

Encourage each component to focus on a subset of links:

```math
\lambda_A \|A\|_1
```

### Smooth Packet/Time Factor

Human motion should not jump randomly:

```math
\lambda_C \sum_{r,p} |C(p+1,r)-C(p,r)|
```

### Diversity/Orthogonality

Reduce duplicate components:

```math
\lambda_D(\|A^TA-I\|_F^2+\|B^TB-I\|_F^2+\|C^TC-I\|_F^2)
```

### Pose-Supervised Component Selection

Add a weak supervised term so components are not only reconstructive but also
pose-informative:

```math
\mathcal{L} =
\mathcal{L}_{rec}
+ \lambda_{pose}\mathcal{L}_{pose}
+ \lambda_D\mathcal{L}_{div}
+ \lambda_S\mathcal{L}_{sparse}
```

This is likely the key improvement over vanilla CP.

## Paper Contribution

The main contribution should be stated as:

> We introduce a physically interpretable CSI decomposition framework that
> factorizes Wi-Fi CSI into compact link, frequency/Doppler, and temporal
> components. These components preserve CSI structure while enabling
> super-lightweight ML backbones to approach neural Wi-Fi HPE accuracy.

Supporting contributions:

1. Frame-level decomposition under the exact HPE-Li MM-Fi protocol.
2. Lightweight ML pose regressors on decomposed components.
3. Decomposition quality metrics: reconstruction, compactness, separation,
   interpretability, and pose-relevance.
4. Fair comparison against HPE-Li on full MM-Fi once data is complete.
5. Constrained/supervised decomposition to make components physically cleaner.

## Immediate Next Steps

1. Complete the missing MM-Fi frames so the evaluation uses all 320,760 samples.
2. Run rank sweep `R = 2, 4, 6, 8, 12`.
3. Add component-ablation experiment.
4. Add constrained decomposition:
   - diversity penalty,
   - sparse factors,
   - smooth packet/time factor.
5. Add basic CNN baseline once PyTorch is installed.
6. Update the paper around decomposition quality, not only PCK.
