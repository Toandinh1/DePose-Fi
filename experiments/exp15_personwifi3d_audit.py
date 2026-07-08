import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]

CSI_HINTS = ("csi", "wifi", "signal", "input", "x", "amp", "phase")
POSE_HINTS = ("pose", "joint", "keypoint", "skeleton", "label", "target", "y", "gt")


def is_hidden(path):
    return any(part.startswith(".") for part in path.parts)


def summarize_array(name, arr):
    arr = np.asarray(arr)
    if arr.dtype == object:
        return {
            "name": name,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "min": "",
            "max": "",
            "finite_pct": "",
        }
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else None
    finite_pct = "" if finite is None else f"{100.0 * finite.mean():.2f}"
    if finite is not None and finite.any():
        values = arr[finite]
        mn = f"{float(values.min()):.6g}"
        mx = f"{float(values.max()):.6g}"
    else:
        mn = ""
        mx = ""
    return {
        "name": name,
        "shape": str(arr.shape),
        "dtype": str(arr.dtype),
        "min": mn,
        "max": mx,
        "finite_pct": finite_pct,
    }


def score_name(name, hints):
    name = name.lower()
    return sum(1 for hint in hints if hint in name)


def inspect_npz(path):
    rows = []
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            rows.append(summarize_array(key, z[key]))
    return rows


def inspect_npy(path):
    arr = np.load(path, allow_pickle=False)
    return [summarize_array(path.stem, arr)]


def inspect_mat(path):
    mat = loadmat(path)
    rows = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        rows.append(summarize_array(key, value))
    return rows


def inspect_file(path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            return inspect_npz(path)
        if suffix == ".npy":
            return inspect_npy(path)
        if suffix == ".mat":
            return inspect_mat(path)
    except Exception as exc:
        return [
            {
                "name": "<inspect_error>",
                "shape": "",
                "dtype": type(exc).__name__,
                "min": "",
                "max": str(exc),
                "finite_pct": "",
            }
        ]
    return []


def likely_role(row):
    name = row["name"]
    csi = score_name(name, CSI_HINTS)
    pose = score_name(name, POSE_HINTS)
    shape = row["shape"]
    if pose > csi:
        return "pose_candidate"
    if csi > pose:
        return "csi_candidate"
    if "17" in shape or "18" in shape or "25" in shape:
        return "pose_candidate"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "PersonInWiFi3D")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "personwifi3d_audit.csv")
    parser.add_argument("--max-files", type=int, default=200)
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(
            f"Dataset folder not found: {args.data_root}\n"
            "Place Person-in-WiFi 3D under this folder, or pass --data-root."
        )

    candidates = []
    for suffix in ("*.npz", "*.npy", "*.mat"):
        candidates.extend(p for p in args.data_root.rglob(suffix) if not is_hidden(p))
    candidates = sorted(candidates)[: args.max_files]

    rows = []
    for path in candidates:
        for arr in inspect_file(path):
            row = {
                "file": str(path.relative_to(args.data_root)),
                "role": likely_role(arr),
                **arr,
            }
            rows.append(row)

    args.output.parent.mkdir(exist_ok=True)
    fieldnames = ["file", "role", "name", "shape", "dtype", "min", "max", "finite_pct"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_csi = sum(row["role"] == "csi_candidate" for row in rows)
    n_pose = sum(row["role"] == "pose_candidate" for row in rows)
    print(f"data_root={args.data_root}")
    print(f"files_inspected={len(candidates)}")
    print(f"arrays_found={len(rows)}")
    print(f"csi_candidates={n_csi}")
    print(f"pose_candidates={n_pose}")
    print(f"saved_audit={args.output}")
    if n_csi == 0 or n_pose == 0:
        print(
            "No reliable CSI/pose pair inferred yet. Send or place the dataset files, "
            "then inspect the audit CSV and map the CSI and pose keys into a canonical NPZ."
        )


if __name__ == "__main__":
    main()
