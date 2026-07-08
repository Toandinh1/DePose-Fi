# Falsification Protocol: Is Tensor-Factorized CSI HPE Viable?

## Purpose

We need to answer this question honestly:

> Is tensor-factorized CSI a useful representation for Wi-Fi HPE, or is it a dead end?

The project should be killed or reframed if CP factors do not improve downstream pose estimation under controlled comparisons.

## Current First Probe

Dataset sample:

- MM-Fi Kaggle sample.
- Sequence: `E01/E01/S01/A01`.
- CSI frames: 297.
- Ground truth: `(297, 17, 3)`.
- CSI frame shape: `CSIamp/CSIphase = (3, 114, 10)`.

Experiment:

- Sliding windows of 32 CSI frames.
- Target is center-frame 3D keypoints.
- Compared:
  - mean-pose baseline,
  - raw link-Doppler-time tensor + Ridge,
  - rank-4 CP factor features + Ridge.

Result:

```text
mean_pose:      MPJPE=0.0840, PCK@0.05=0.087, PCK@0.10=0.720, PCK@0.20=1.000
raw_ldt_ridge:  MPJPE=0.0839, PCK@0.05=0.098, PCK@0.10=0.725, PCK@0.20=1.000
cp_rank4_ridge: MPJPE=0.0843, PCK@0.05=0.106, PCK@0.10=0.720, PCK@0.20=1.000
```

Interpretation:

This does **not** verify the idea. The sequence is too easy/static because mean-pose already performs almost as well as learned features and PCK@0.20 is saturated.

This also does **not** kill the idea. It only says one short single-action sequence is not a valid test.

## Required Falsification Ladder

### Test 1: Alignment Sanity Check

Goal: verify CSI windows are synchronized with pose labels.

Run:

- same-action chronological split,
- random split,
- shuffled-label control.

Pass condition:

- real labels must beat shuffled labels.
- raw CSI/Doppler must beat mean pose on at least one non-saturated metric.

Fail implication:

- data alignment or preprocessing is wrong, or this modality/action has too little pose variation.

### Test 2: Multi-Action Within-Subject Test

Use multiple actions:

```text
E01/E01/S01/A01 ... A14
```

Train/test options:

- train on 70% windows pooled across actions, test on 30%.
- harder: leave-one-action-out.

Baselines:

1. mean pose,
2. raw dynamic CSI,
3. raw link-Doppler-time tensor,
4. matrix NMF,
5. CP factors.

Pass condition:

- CP beats raw LDT or NMF on MPJPE or keypoint-wise PCK.
- improvement should appear on high-motion joints: wrists, elbows, knees, ankles.

Fail implication:

- CP may not add useful representation beyond raw Doppler features.

### Test 3: Rank and Constraint Sweep

Sweep:

```text
R = 2, 4, 6, 8, 10, 12
```

Compare:

- unconstrained CP,
- smooth temporal CP,
- sparse temporal CP,
- smooth + sparse CP.

Pass condition:

- there exists a stable rank/constraint setting that improves HPE, not just reconstruction.

Fail implication:

- factorization is either too lossy or not aligned with pose.

### Test 4: Matrix NMF vs Tensor CP

This is paper-critical.

Pass condition:

- tensor CP beats matrix NMF under the same regression/network.

Fail implication:

- preserving tensor structure is not giving measurable benefit, weakening the main novelty.

### Test 5: Multi-Person / Confusion Test

Only run after single-person/multi-action test works.

Pass condition:

- CP reduces keypoint confusion or improves high-motion keypoints in two-person scenes.

Fail implication:

- do not make multi-person claims.

## Kill Criteria

The direction is likely not publishable in current form if all are true:

1. Raw LDT consistently beats or matches CP across actions.
2. Matrix NMF matches CP.
3. CP reconstruction improves but pose accuracy does not.
4. CP factors are unstable across seeds/ranks.
5. Improvements disappear under shuffled/random controls.

If this happens, reframe the paper away from CP factorization and toward learned/differentiable decomposition.

## Green-Light Criteria

The direction is promising if:

1. CP beats mean pose and raw LDT on non-saturated metrics.
2. CP beats matrix NMF.
3. Gains concentrate on high-motion keypoints.
4. Factors show interpretable Doppler/time patterns.
5. Results hold across multiple actions, not only one sequence.

## Next Experiment

Download a small multi-action subset:

```text
A01-A05, first 120 or all 297 Wi-Fi CSI frames per action
```

Then run:

1. multi-action data loader,
2. raw LDT vs CP Ridge probe,
3. shuffled-label control,
4. leave-one-action-out split.
