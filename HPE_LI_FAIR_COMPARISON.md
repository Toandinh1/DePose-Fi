# Fair Comparison Against HPE-Li

## Problem With the Current Probe

The current CP probe is not directly comparable to HPE-Li because it builds a
link-Doppler-time tensor from a sliding temporal window. For example, a 32-frame
window uses 32 MM-Fi frames, and each frame contains 10 short CSI packets. That
means the probe can use about 320 short-time CSI packets to predict one pose.

HPE-Li, in contrast, uses MM-Fi with `data_unit: frame`. Each training sample is
one `frameXXX.mat` file, whose Wi-Fi CSI input is `CSIamp` with shape:

```text
3 x 114 x 10
```

Therefore, HPE-Li predicts each pose from one frame-level CSI sample, not from a
long temporal context.

## HPE-Li MM-Fi Protocol

From `HPE-Li-ECCV2024/dataset_lib/config.yaml` and
`HPE-Li-ECCV2024/dataset_lib/mmfi.py`:

- modality: `wifi-csi`
- protocol: `protocol3`
- data unit: `frame`
- random split:
  - ratio: `0.7`
  - seed: `0`
- frame count per action: `297`
- actions under protocol3: `A01` to `A27`
- subjects: `S01` to `S40`
- each frame-level Wi-Fi sample is normalized independently to `[0, 1]`
- validation split is further split into validation/test by
  `train_test_split(..., test_size=0.5, random_state=41)`

The PCK metric uses 2D keypoints and normalizes joint error by the distance
between MM-Fi keypoints 1 and 11.

## Fair Comparison Options

### Option A: Single-Frame Fairness

Every method receives only one MM-Fi frame:

```text
input = CSIamp(frame_t), shape 3 x 114 x 10
target = pose(frame_t)
```

Compare:

1. HPE-Li original.
2. Raw CSI + same HPE-Li backbone.
3. Single-frame factorized/decomposed representation + same backbone.
4. DePose-Fi single-frame variant.

This is the cleanest comparison to HPE-Li, but it weakens Doppler/tensor
factorization because 10 packets provide limited temporal structure.

### Option B: Equal-Window Fairness

Every method receives the same temporal window around frame `t`:

```text
input = frames [t-w, ..., t+w]
target = pose(frame_t)
```

Compare:

1. Temporal HPE-Li baseline on the same window.
2. Raw link-Doppler-time tensor model.
3. NMF-decomposed tensor model.
4. CP-decomposed tensor model.
5. DePose-Fi final model.

This tests the decomposition idea more honestly, but we cannot compare directly
to published HPE-Li numbers unless we retrain HPE-Li with the same window.

## Recommendation

Use both protocols:

1. **Published-baseline comparison:** report HPE-Li's published single-frame
   results and reproduce its frame-level split as closely as possible.
2. **Controlled fair ablation:** give HPE-Li, raw LDT, NMF, CP, and DePose-Fi the
   same temporal window and compare them under identical train/test splits.

The paper should not claim superiority over HPE-Li using a temporal-window
method unless HPE-Li is also given the same temporal-window input.

## Immediate Implementation Tasks

1. Build an HPE-Li-compatible MM-Fi frame manifest using the same protocol3,
   random split, and test split.
2. Recompute `mean_pose` on that split as a sanity check.
3. Implement a single-frame DePose-Fi variant, even if it is weak.
4. Implement a temporal-window HPE-Li baseline for the controlled comparison.
5. Report PCK_50, PCK_40, PCK_30, PCK_20, PCK_10, MPJPE, PA-MPJPE, parameter
   count, FLOPs, and latency.
