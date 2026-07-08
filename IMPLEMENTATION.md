# Implementation Roadmap

## Goal

Build the minimum real experiment needed to test the paper claim:

> Tensor-factorized link-Doppler-time CSI components improve Wi-Fi HPE compared with raw dynamic CSI, raw Doppler tensors, and matrix NMF features.

## Required Project Structure

```text
Decompose_WiFi_Sensing_HPE/
  data/                         # not committed; real datasets or symlinks
  src/
    data_loading.py             # load CSI and pose labels
    preprocessing.py            # static removal, CSI reshaping
    doppler_tensor.py           # STFT and link-Doppler-time construction
    cp_factorization.py         # non-negative CP, constraints, reconstruction
    nmf_baseline.py             # matrix NMF baseline
    heatmap.py                  # keypoint-to-heatmap conversion
    pose_network.py             # simple shared CNN heatmap estimator
    metrics.py                  # PCK, MPJPE, factor reconstruction error
  experiments/
    exp01_synthetic_cp.py
    exp02_build_tensors.py
    exp03_factorization_sweep.py
    exp04_single_person_hpe.py
    exp05_nmf_vs_cp.py
    exp06_two_person_hpe.py
  PAPER/
    main.tex
```

## Stage 0: Current Files

Already created:

- `idea.pdf`: original proposal.
- `idea.txt`: extracted text from the proposal.
- `publication_plan.md`: paper direction, math, experiments, risks.
- `synthetic_cp_probe.py`: synthetic CP sanity check.
- `synthetic_cp_probe.png`: visualization from synthetic probe.
- `PAPER/main.tex`: first paper draft.

Current synthetic CP result:

```text
relative_reconstruction_error = 0.2212
mean_matched_link_correlation = 0.869
mean_matched_doppler_correlation = 0.871
mean_matched_time_correlation = 0.857
```

Current MM-Fi Kaggle sample status:

- Kaggle authentication should be configured locally with the user's own `kaggle.json`.
- Downloaded `E01/E01/S01/A01/ground_truth.npy`.
- Downloaded first 120 Wi-Fi CSI frames from `E01/E01/S01/A01/wifi-csi`.
- One CSI frame contains:
  - `CSIamp` with shape `(3, 114, 10)`.
  - `CSIphase` with shape `(3, 114, 10)`.
- Concatenating 120 frames gives complex CSI `H` with shape `(3, 114, 1200)`.
- Ground truth has shape `(297, 17, 3)`, apparently 17 3D keypoints.
- First real link-Doppler-time tensor generated with shape `(3, 64, 72)`.
- Rank-4 CP on this tensor gives relative reconstruction error `0.5305`.
- Rank sweep results:
  - `R=1`: `0.6550`
  - `R=2`: `0.6062`
  - `R=3`: `0.5654`
  - `R=4`: `0.5306`
  - `R=5`: `0.5017`
  - `R=6`: `0.4765`
  - `R=7`: `0.4528`
  - `R=8`: `0.4294`
  - `R=9`: `0.4143`
  - `R=10`: `0.3969`

Created real-data scripts:

- `src/mmfi_pipeline.py`
- `src/cp_factorization.py`
- `experiments/exp01_build_ldt_tensor.py`
- `experiments/exp02_factorize_ldt.py`
- `experiments/exp03_cp_rank_sweep.py`
- `experiments/exp04_pose_regression_probe.py`
- `experiments/exp05_leave_one_action_probe.py`

Full-dataset download attempt:

- `kaggle datasets download ... --unzip` timed out after 20 minutes.
- It left `data/MMFi_full/mmfi-dataset.zip` at about `44.05 GB`, but the archive is truncated/corrupt.
- Streaming extraction from the partial archive recovered a useful Wi-Fi/ground-truth subset:
  - `178,497` `.mat` Wi-Fi CSI files.
  - `601` sequences with both `wifi-csi` and `ground_truth.npy`.
  - extracted size about `9.18 GB`.
- Manifest saved to `outputs/mmfi_extracted_manifest.csv`.

First leave-one-action feasibility result:

- Train: `E01/E01/S01/A01-A04`.
- Test: held-out `E01/E01/S01/A05`.
- 48 windows per action.

```text
mean_pose:      MPJPE=0.1274, PCK@0.05=0.065, PCK@0.10=0.620, PCK@0.20=0.811
raw_ldt_ridge:  MPJPE=0.1530, PCK@0.05=0.039, PCK@0.10=0.464, PCK@0.20=0.768
cp_rank4_ridge: MPJPE=0.1312, PCK@0.05=0.071, PCK@0.10=0.624, PCK@0.20=0.798
```

Interpretation:

- CP factors are much better than raw LDT under this ridge probe.
- CP is still slightly worse than mean-pose in MPJPE.
- This means CP is a useful regularized representation, but current rank-4 offline CP + Ridge is not enough to claim HPE improvement.
- Next: rank sweep for the regression task, smooth/sparse temporal CP, matrix NMF comparison, and stronger splits.

## Stage 1: Data Loader

Inputs needed:

- CSI array with shape compatible with:

```text
Nt x Nr x Nf x T
```

or already merged:

```text
L x Nf x T
```

- pose labels as 2D/3D keypoints.
- sample timestamps or synchronized CSI-pose windows.

Output:

```python
sample = {
    "csi": complex_array,        # L x Nf x T
    "keypoints": float_array,    # J x 2 or J x 3
    "meta": dict
}
```

## Stage 2: Static Removal

Implement:

```math
\bar{H}(l,f)=\frac{1}{T}\sum_{t=1}^{T}H(l,f,t)
```

```math
\Delta H(l,f,t)=H(l,f,t)-\bar{H}(l,f)
```

Outputs:

- raw amplitude plot,
- dynamic amplitude plot,
- before/after energy summary.

## Stage 3: Link-Doppler-Time Tensor

Implement:

```math
X(l,d,\tau)=\sum_{f=1}^{N_f}|\mathrm{STFT}_t(\Delta H(l,f,t))|^2
```

Hyperparameters:

- STFT window length,
- hop length,
- number of Doppler bins,
- whether to use amplitude, complex CSI, phase-corrected CSI, or CSI ratio.

First version:

- use dynamic CSI power,
- aggregate over subcarriers,
- output `X` with shape `L x D x T_prime`.

## Stage 4: CP Factorization

Implement non-negative CP:

```math
X(l,d,\tau)\approx \sum_{r=1}^{R}A(l,r)B(d,r)C(\tau,r)
```

Sweep:

```text
R = 2, 4, 6, 8, 12
```

Report:

- normalized reconstruction error,
- factor visualizations,
- runtime per sample,
- downstream pose accuracy.

Important: do not select `R` only by reconstruction error.

## Stage 5: NMF Baseline

Unfold the tensor into a matrix:

```text
L x (D*T_prime)
```

or:

```text
(L*D) x T_prime
```

Apply non-negative matrix factorization:

```math
X_{matrix}\approx WH
```

Then reconstruct NMF component tensors in the closest possible format to CP components.

Purpose:

Show that preserving tensor structure matters.

## Stage 6: Pose Heatmap Network

Use the same simple network for every input representation.

Inputs:

1. raw dynamic CSI power,
2. raw link-Doppler-time tensor,
3. matrix NMF components,
4. CP tensor components.

Output:

```text
J x H x W heatmaps
```

Loss:

```math
\mathcal{L}_{pose}=\sum_j\|\hat{Y}_j-Y_j\|_2^2
```

## Stage 7: Metrics

Pose:

- PCK@5,
- PCK@10,
- PCK@20,
- MPJPE if coordinates are decoded,
- keypoint-wise PCK.

Factorization:

- normalized reconstruction error,
- factor stability over seeds,
- visualization of `A`, `B`, `C`,
- temporal activation alignment with motion periods.

## Stage 8: Paper-Critical Ablations

Must-have:

1. Raw Doppler vs CP.
2. Matrix NMF vs CP.
3. Vary `R`.
4. With/without smoothness.
5. With/without sparsity.
6. Single-person first.
7. Two-person only after single-person works.

## Decision Rules

The idea is worth pushing if:

- CP beats raw Doppler on average PCK or difficult keypoints,
- CP beats NMF under the same network,
- the improvement is not only from increased input channels,
- factor visualizations are interpretable enough for a paper figure.

The idea is weak if:

- CP reconstruction error improves but HPE accuracy does not,
- raw Doppler + CNN matches or beats CP,
- NMF performs the same as CP,
- factors are unstable across random seeds.

## Next Immediate Task

We need the real dataset path.

If real data is unavailable, run extended synthetic experiments:

1. noise sweep,
2. rank sweep,
3. two-person mixture simulation,
4. CP vs NMF factor recovery,
5. synthetic pose-regression from raw tensor vs CP features.
