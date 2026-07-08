from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import nonnegative_cp_mu  # noqa: E402
from mmfi_pipeline import (  # noqa: E402
    build_link_doppler_time_tensor,
    load_wifi_csi_frames,
    normalize_tensor,
    remove_static_component,
)


def mpjpe(y_true, y_pred):
    y_true = y_true.reshape((-1, 17, 3))
    y_pred = y_pred.reshape((-1, 17, 3))
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=2)))


def pck(y_true, y_pred, threshold):
    y_true = y_true.reshape((-1, 17, 3))
    y_pred = y_pred.reshape((-1, 17, 3))
    err = np.linalg.norm(y_true - y_pred, axis=2)
    return float(np.mean(err < threshold))


def cp_feature(x, rank=4, iters=250):
    a, b, c, _ = nonnegative_cp_mu(x, rank=rank, iters=iters, seed=rank)
    # Normalize away arbitrary CP scale for regression stability.
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()])


def main():
    frame_dir = ROOT / "data" / "MMFi_sample" / "E01_E01_S01_A01_wifi"
    gt_path = ROOT / "data" / "MMFi_sample" / "ground_truth.npy"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    h, paths = load_wifi_csi_frames(frame_dir, limit=297)
    gt = np.load(gt_path).astype(np.float64)
    if len(paths) != gt.shape[0]:
        raise ValueError(f"CSI frame count {len(paths)} != GT frame count {gt.shape[0]}")

    window_frames = 32
    stride = 4
    half = window_frames // 2
    centers = list(range(half, len(paths) - half, stride))

    raw_features = []
    cp_features = []
    targets = []
    used_centers = []

    for idx, center in enumerate(centers):
        start = center - half
        end = center + half
        packet_start = start * 10
        packet_end = end * 10

        h_win = h[:, :, packet_start:packet_end]
        delta, _ = remove_static_component(h_win)
        x, _, _ = build_link_doppler_time_tensor(delta, nperseg=64, noverlap=48, nfft=64)
        x = normalize_tensor(x)

        raw_features.append(x.ravel())
        cp_features.append(cp_feature(x, rank=4, iters=250))
        targets.append(gt[center].ravel())
        used_centers.append(center)
        if (idx + 1) % 10 == 0:
            print(f"processed_windows={idx + 1}/{len(centers)}")

    x_raw = np.vstack(raw_features)
    x_cp = np.vstack(cp_features)
    y = np.vstack(targets)
    centers = np.asarray(used_centers)

    split = int(0.7 * len(y))
    train = np.arange(split)
    test = np.arange(split, len(y))
    alphas = np.logspace(-3, 3, 13)

    mean_pred = np.repeat(y[train].mean(axis=0, keepdims=True), len(test), axis=0)
    models = {
        "mean_pose": None,
        "raw_ldt_ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        "cp_rank4_ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
    }

    results = {}
    for name, model in models.items():
        if name == "mean_pose":
            pred = mean_pred
        else:
            x_feat = x_raw if name == "raw_ldt_ridge" else x_cp
            model.fit(x_feat[train], y[train])
            pred = model.predict(x_feat[test])
        results[name] = {
            "mpjpe": mpjpe(y[test], pred),
            "pck_005": pck(y[test], pred, 0.05),
            "pck_010": pck(y[test], pred, 0.10),
            "pck_020": pck(y[test], pred, 0.20),
        }

    for name, res in results.items():
        print(
            f"{name}: MPJPE={res['mpjpe']:.4f} "
            f"PCK@0.05={res['pck_005']:.3f} "
            f"PCK@0.10={res['pck_010']:.3f} "
            f"PCK@0.20={res['pck_020']:.3f}"
        )

    out_npz = out_dir / "E01_E01_S01_A01_pose_probe.npz"
    np.savez_compressed(
        out_npz,
        centers=centers,
        X_raw=x_raw,
        X_cp=x_cp,
        y=y,
        split=split,
        **{f"{k}_{m}": v[m] for k, v in results.items() for m in v},
    )

    names = list(results)
    mpjpes = [results[n]["mpjpe"] for n in names]
    pck20 = [results[n]["pck_020"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(names, mpjpes)
    axes[0].set_ylabel("MPJPE (lower is better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, pck20)
    axes[1].set_ylabel("PCK@0.20 (higher is better)")
    axes[1].tick_params(axis="x", rotation=20)
    out_png = out_dir / "E01_E01_S01_A01_pose_probe.png"
    fig.savefig(out_png, dpi=180)
    print(f"samples={len(y)} train={len(train)} test={len(test)}")
    print(f"raw_feature_dim={x_raw.shape[1]} cp_feature_dim={x_cp.shape[1]}")
    print(f"saved_npz={out_npz}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
