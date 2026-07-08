# SwiftPose-Fi / Decomposition-Based Wi-Fi HPE

This repository contains research code and paper drafts for a decomposition-first Wi-Fi human pose estimation project.

The main idea is to decompose Wi-Fi CSI into structured components using CP factorization, then estimate pose with a lightweight Selective Component-Adaptive Fusion (S-AFF) model.

## Contents

- `src/`: shared CP factorization, MM-Fi loading utilities, and metrics.
- `experiments/`: experiment scripts for MM-Fi and Person-in-WiFi 3D studies.
- `PAPER/`: LaTeX draft and generated figures.
- `data/`: dataset notes only. Raw datasets are not included.

## Main Experiments

MM-Fi full protocol-3 frame-level S-AFF:

```bash
python experiments/exp14_cp_cnn_aff.py
```

Person-in-WiFi 3D mixed-person CP + query S-AFF:

```bash
python experiments/exp17_piw3d_cp_saff.py --data-root D:/TinySense/PiW_dataset
```

Person-in-WiFi 3D single-person dual amplitude/phase CP streams:

```bash
python experiments/exp19_piw3d_dualcp_saff.py --data-root D:/TinySense/PiW_dataset
```

## Data

Datasets are intentionally not committed.

- MM-Fi should be placed under `data/MMFi_full/`.
- Person-in-WiFi 3D should be placed locally and passed with `--data-root`, for example `D:/TinySense/PiW_dataset`.

See `data/PersonInWiFi3D_README.md` for the expected Person-in-WiFi 3D layout.

## Notes

Generated feature caches, trained checkpoints, external cloned repositories, and reference PDFs are excluded from Git to keep the repository lightweight and shareable.
