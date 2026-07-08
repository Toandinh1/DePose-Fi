# Person-in-WiFi 3D Evaluation Notes

The Person-in-WiFi 3D dataset is not currently present in this workspace.
Place the dataset under:

```text
data/PersonInWiFi3D/
```

Then audit the available files:

```powershell
python experiments/exp15_personwifi3d_audit.py --data-root data/PersonInWiFi3D
```

The cross-dataset runner expects a canonical NPZ file with:

```text
wifi: N x L x S x P
pose: N x J x 3 or N x M x J x 3
```

For multi-person labels, the current first-pass protocol uses the first annotated
person per frame:

```powershell
python experiments/exp16_canonical_cp_aff.py `
  --input outputs/personwifi3d_canonical.npz `
  --output outputs/personwifi3d_cp_aff.csv `
  --rank 4 `
  --cp-iters 10 `
  --epochs 40
```

Do not report Person-in-WiFi 3D results in the paper until the canonical file is
built from the official dataset and the split/metric are documented.
