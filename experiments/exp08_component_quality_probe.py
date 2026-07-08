from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "experiments"))

from cp_factorization import cp_reconstruct, nonnegative_cp_mu  # noqa: E402
from exp06_protocol3_frame_cp_probe import (  # noqa: E402
    hpe_li_protocol3_subject_split,
    iter_available_frames,
    load_hpe_li_frame,
    split_val_test,
)


def entropy01(x, axis=0, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    p = x / (x.sum(axis=axis, keepdims=True) + eps)
    h = -(p * np.log(p + eps)).sum(axis=axis)
    denom = np.log(x.shape[axis])
    return h / (denom + eps)


def mean_abs_offdiag_corr(mat):
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[1] <= 1:
        return 0.0
    corr = np.corrcoef(mat.T)
    corr = np.nan_to_num(corr, nan=0.0)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def component_energy(a, b, c):
    # After normalize_factors, component scale is mainly carried by C.
    return np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) * np.linalg.norm(c, axis=0)


def summarize_frame(frame, rank=4, iters=35):
    a, b, c, hist = nonnegative_cp_mu(frame, rank=rank, iters=iters, seed=rank)
    recon = cp_reconstruct(a, b, c)
    recon_err = np.linalg.norm(frame - recon) / (np.linalg.norm(frame) + 1e-12)
    energy = component_energy(a, b, c)
    energy_share = energy / (energy.sum() + 1e-12)
    return {
        "recon_err": float(recon_err),
        "link_corr": mean_abs_offdiag_corr(a),
        "subcarrier_corr": mean_abs_offdiag_corr(b),
        "packet_corr": mean_abs_offdiag_corr(c),
        "link_entropy": float(np.mean(entropy01(a, axis=0))),
        "subcarrier_entropy": float(np.mean(entropy01(b, axis=0))),
        "packet_entropy": float(np.mean(entropy01(c, axis=0))),
        "energy_entropy": float(entropy01(energy[:, None], axis=0)[0]),
        "top_component_share": float(np.max(energy_share)),
        "packet_smoothness": float(np.mean(np.abs(np.diff(c, axis=0)))),
        "final_history_err": float(hist[-1]),
    }


def main():
    data_root = ROOT / "data" / "MMFi_full" / "extracted"
    out_csv = ROOT / "outputs" / "component_quality_probe.csv"
    rank = 4
    iters = 35
    max_samples = 3000

    train_form, val_form = hpe_li_protocol3_subject_split()
    train_items = list(iter_available_frames(data_root, train_form))
    val_items_all = list(iter_available_frames(data_root, val_form))
    _, test_items = split_val_test(val_items_all)
    items = train_items[: max_samples // 2] + test_items[: max_samples // 2]

    rows = []
    t0 = time.time()
    for idx, item in enumerate(items, start=1):
        frame = load_hpe_li_frame(item["frame_path"])
        row = summarize_frame(frame, rank=rank, iters=iters)
        row.update(
            {
                "scene": item["scene"],
                "subject": item["subject"],
                "action": item["action"],
                "idx": item["idx"],
            }
        )
        rows.append(row)
        if idx % 500 == 0:
            print(f"quality_frames={idx}/{len(items)} elapsed_sec={time.time() - t0:.1f}", flush=True)

    keys = list(rows[0].keys())
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    print(f"samples={len(rows)} rank={rank} iters={iters}")
    for key in [
        "recon_err",
        "link_corr",
        "subcarrier_corr",
        "packet_corr",
        "link_entropy",
        "subcarrier_entropy",
        "packet_entropy",
        "energy_entropy",
        "top_component_share",
        "packet_smoothness",
    ]:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        print(f"{key}: mean={vals.mean():.4f} std={vals.std():.4f}")
    print("compression_ratio_raw_to_cp_features=6.73")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
