# DePose-Fi

This repository contains the research code and paper draft for a decomposition-first Wi-Fi human pose estimation project.

## Project Goal

Wi-Fi human pose estimation can work without cameras, but many existing methods directly feed raw channel state information (CSI) into large neural networks. That gives good accuracy, but it is difficult to deploy on small edge hardware.

This project asks a different question:

> Can we first decompose CSI into interpretable wireless components, then use a much smaller pose regressor that is easier to deploy?

Our main pipeline is:

1. Convert Wi-Fi CSI into a tensor.
2. Decompose the tensor using CP factorization.
3. Treat the CP factors as interpretable components:
   - link / antenna-pair component,
   - subcarrier / frequency component,
   - packet / time component.
4. Feed these components into **Selective Component-Adaptive Fusion (S-AFF)**.
5. Predict human pose with a lightweight model built from standard operations.

## Main Takeaway

The strongest result is on **MM-Fi**, where the decomposition + S-AFF design gives a much smaller model while staying close to a strong Wi-Fi HPE baseline.

For a compact stakeholder-facing summary of current outcomes, see:

```text
RESULTS_SUMMARY.md
```

Current headline result:

| Dataset / Setting | Method | Accuracy | Params | FLOPs |
|---|---:|---:|---:|---:|
| MM-Fi protocol-3 frame-level | HPE-Li baseline | 52.07 PCK20 | 1.66M | 2.42G |
| MM-Fi protocol-3 frame-level | CP + S-AFF | 50.80 PCK20 | 64.9K | 2.89M |

So the MM-Fi story is:

- only 1.27 PCK20 behind HPE-Li,
- about 26x fewer parameters,
- more than 800x fewer FLOPs,
- standard operations only, which is better for edge deployment.

## Why S-AFF?

Plain fusion averages or concatenates all CP factors. That is weak because different frames may rely on different CSI components.

S-AFF improves this by keeping branch-specific embeddings:

- `h_A`: link/antenna factor embedding,
- `h_B`: subcarrier factor embedding,
- `h_C`: packet factor embedding,
- `h_F`: fused embedding.

The model predicts a gate over these experts and can emphasize the most useful branch for each sample. We also use a sharpening loss so the gate avoids nearly uniform weights.

Important distinction:

- Current S-AFF is a **soft fusion** model. It computes all branches and then weights them.
- A future hard-routing version could compute only the selected branch to reduce real inference cost further.

## Person-in-WiFi 3D Adaptation

Person-in-WiFi 3D is harder than MM-Fi because it is 3D and can contain multiple people. The official format has CSI shaped as `3 x 3 x 30 x 20`, which we reshape into antenna links, subcarriers, and packets.

We learned two important things:

1. A fixed one-person head is not enough for multi-person data.
2. For Person-in-WiFi 3D-style data, amplitude and phase should be decomposed with **separate CP streams** instead of being mixed into one tensor.

Current single-person Person-in-WiFi 3D result:

| Method | MPJPE | Params |
|---|---:|---:|
| WiFi-Mamba SOTA | 76.75 mm | 2.14M |
| Our dual amplitude/phase CP + S-AFF-L | 83.82 mm | 1.70M |
| Our dual amplitude/phase CP + S-AFF-M | 91.35 mm | 923K |
| Our dual amplitude/phase CP + S-AFF-S | 107.53 mm | 248K |

This is not yet a SOTA accuracy win. The honest interpretation is:

- WiFi-Mamba is still better in accuracy.
- Our large model is close but does not give a big parameter advantage.
- Our medium and small variants provide better deployment tradeoffs.
- Separate amplitude/phase CP streams are the right direction for improving the PiW setting.

## Repository Structure

```text
.
|-- src/
|   |-- cp_factorization.py      # CP decomposition helpers
|   |-- metrics.py               # Pose metrics
|   `-- mmfi_pipeline.py         # MM-Fi dataset utilities
|-- experiments/
|   |-- exp14_cp_cnn_aff.py      # Main MM-Fi CP + CNN/AFF/S-AFF experiment
|   |-- exp17_piw3d_cp_saff.py   # Person-in-WiFi 3D mixed-person CP + query S-AFF
|   |-- exp18_piw3d_temporal_cp_saff.py
|   |                            # Temporal CP embedding experiment
|   |-- exp19_piw3d_dualcp_saff.py
|   |                            # Dual amplitude/phase CP streams for PiW
|   |-- exp20_saff_parallel_inference.py
|                                # S-AFF branch/stream parallel inference benchmark
|   |-- exp21_saff_onnx_parallel.py
|                                # ONNX Runtime deployment and split-stream benchmark
|   |-- exp22_hpe_li_runtime.py
|   |                            # HPE-Li PyTorch/ONNX runtime benchmark
|   |-- exp23_mmfi_saff_runtime.py
|                                # MM-Fi CP + S-AFF PyTorch/ONNX runtime benchmark
|   |-- exp24_hard_routed_saff.py
|   |                            # Hard-routed S-AFF accuracy/compute tradeoff
|   |-- exp25_mmfi_bonly_runtime.py
|                                # Subcarrier-only routed runtime benchmark
|   |-- exp26_decomposition_feature_comparison.py
|                                # PCA/NMF/Tucker/CP feature comparison
|   `-- exp27_decomposition_regressor_ablation.py
|                                # PCA/NMF/Tucker/CP with neural regressors
|-- PAPER/
|   |-- deposefi_systems_draft.tex
|   `-- figures/
|-- data/
|   `-- PersonInWiFi3D_README.md # Dataset layout notes only
`-- README.md
```

## Setup

Create a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The code expects common scientific Python packages:

- `numpy`
- `scipy`
- `scikit-learn`
- `torch`
- `h5py`
- `matplotlib`

## Datasets

Raw datasets are not committed to GitHub.

Expected local paths:

```text
data/MMFi_full/
data/PersonInWiFi3D/
```

You can also pass a custom Person-in-WiFi 3D path:

```bash
python experiments/exp17_piw3d_cp_saff.py --data-root /path/to/PiW_dataset
```

For Person-in-WiFi 3D, see:

```text
data/PersonInWiFi3D_README.md
```

## Reproducing Main Experiments

### MM-Fi: CP + S-AFF

This is the main lightweight deployment result.

```bash
python experiments/exp14_cp_cnn_aff.py
```

This script compares CP-based regressors, including CNN, AFF, and S-AFF.

### Person-in-WiFi 3D: Mixed-Person Query S-AFF

```bash
python experiments/exp17_piw3d_cp_saff.py \
  --data-root data/PersonInWiFi3D
```

This version handles variable-person labels with a query-style pose head and matching-based evaluation.

### Person-in-WiFi 3D: Dual Amplitude/Phase CP Streams

```bash
python experiments/exp19_piw3d_dualcp_saff.py \
  --data-root data/PersonInWiFi3D
```

This is the most important PiW direction. It decomposes amplitude and phase separately, then fuses them.

### S-AFF Parallel Inference Benchmark

This checks whether branch-level parallelism gives real CPU latency gains before testing real edge devices.

```bash
python experiments/exp20_saff_parallel_inference.py \
  --model-sizes small,medium,large \
  --torch-threads 1,2,4 \
  --warmup 100 \
  --iters 1000
```

Current result: Python-thread branch/stream parallelism is slower than sequential batch-1 inference. The architecture exposes parallelism, but we should not claim measured speedup until ONNX/C++/real-device tests show it.

### ONNX Runtime Deployment Benchmark

This exports S-AFF as both a full ONNX graph and split amplitude/phase stream ONNX graphs.

```bash
python experiments/exp21_saff_onnx_parallel.py \
  --model-size large \
  --warmup 100 \
  --iters 1000 \
  --intra-threads 1,2,4 \
  --inter-threads 1,2 \
  --execution-modes sequential,parallel \
  --rebuild
```

Current result: ONNX Runtime is a major deployment win, giving about 4.5x to 8x faster batch-1 inference than PyTorch on laptop CPU. Split stream execution works, but is not faster than monolithic ONNX on this CPU.

### HPE-Li Runtime Comparison

This compares the local HPE-Li MM-Fi model against our MM-Fi CP + S-AFF runtime.

```bash
python experiments/exp22_hpe_li_runtime.py
python experiments/exp23_mmfi_saff_runtime.py
```

Current result: CP + S-AFF is about 70x faster than HPE-Li in PyTorch CPU inference and about 83x faster in ONNX Runtime CPU inference on this laptop benchmark.

### Hard-Routed S-AFF Deployment

This tests whether the learned S-AFF gate can become a deployment-time routing policy.

```bash
python experiments/exp24_hard_routed_saff.py
python experiments/exp25_mmfi_bonly_runtime.py
```

Current result: the trained MM-Fi S-AFF gate selects the subcarrier expert for 100% of test frames. Top-1 routed inference preserves the full model's PCK20 in this run while reducing ONNX latency from 86.77 us to 54.18 us.

### Decomposition Feature Comparison

This compares PCA, matrix-NMF, Tucker, and CP features using matched neural regressors. The full run uses the official MM-Fi protocol-3 split from `D:\TinySense\MM-Fi`.

```bash
python experiments/exp26_decomposition_feature_comparison.py
python experiments/exp27_decomposition_regressor_ablation.py
```

Current full-MM-Fi result:

| Feature | Regressor | Params | MPJPE | PCK20 |
|---|---|---:|---:|---:|
| PCA | MLP | 111.9K | 0.1872 | 33.53 |
| Matrix-NMF | MLP | 111.9K | 0.2569 | 32.85 |
| Tucker | MLP | 220.7K | 0.1984 | 39.64 |
| CP | MLP | 209.2K | 0.1912 | 42.09 |
| CP | S-AFF | 64.9K | 0.1972 | 50.30 |


## What We Tried and Learned

### Works Well

- CP decomposition gives compact and interpretable CSI components.
- S-AFF improves over plain AFF on MM-Fi.
- Gate sharpening helps avoid uniform fusion weights.
- Query-style heads are necessary for multi-person Person-in-WiFi 3D.
- Separate amplitude/phase CP streams improve Person-in-WiFi 3D single-person accuracy.
- S-AFF exposes branch-level parallelism, but the first Python-thread benchmark shows no batch-1 latency gain yet.

### Did Not Work Well Yet

- Naive temporal modeling over CP vectors hurt PiW performance.
- Simple averaging/smoothing of CP embeddings is not enough.
- Combined amplitude+phase CP is weaker than separate amplitude/phase CP streams.
- PiW multi-person accuracy is still not competitive with heavy SOTA models.
- Naive Python-thread branch parallelism is slower than sequential S-AFF for batch-1 CPU inference.

## Paper Draft

Main draft:

```text
PAPER/deposefi_systems_draft.tex
```

Current framing:

- MM-Fi is the main hardware-friendly result.
- Person-in-WiFi 3D is a generalization/stress-test setting.
- We should not claim SOTA on PiW yet.
- The PiW lesson is that separate amplitude/phase CP streams are required for stronger 3D performance.

## Git Hygiene

Do not commit:

- raw datasets,
- cached `.npz` features,
- trained checkpoints,
- external cloned repositories,
- reference PDFs,
- local absolute paths,
- API keys or tokens.

These are excluded by `.gitignore`.

## Suggested Next Steps

1. Run real edge-device latency tests on Raspberry Pi or Jetson.
2. Export S-AFF to ONNX and compare deployment friction against Mamba-style models.
3. Improve PiW multi-person dual-stream training.
4. Test hard-routing S-AFF for actual conditional computation savings.
5. Add clean result tables and command logs for reproducibility.
