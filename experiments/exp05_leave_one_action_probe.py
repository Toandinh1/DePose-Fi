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
from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402
from mmfi_pipeline import (  # noqa: E402
    build_link_doppler_time_tensor,
    load_wifi_csi_frames,
    normalize_tensor,
    remove_static_component,
)


def cp_feature(x, rank=4, iters=180):
    a, b, c, _ = nonnegative_cp_mu(x, rank=rank, iters=iters, seed=rank)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()])


def sequence_windows(seq_dir, window_frames=32, stride=4, max_windows=None):
    gt = np.load(seq_dir / "ground_truth.npy").astype(np.float64)
    h, paths = load_wifi_csi_frames(seq_dir / "wifi-csi", limit=len(gt))
    if len(paths) != gt.shape[0]:
        raise ValueError(f"{seq_dir}: CSI frame count {len(paths)} != GT frame count {gt.shape[0]}")

    half = window_frames // 2
    centers = list(range(half, len(paths) - half, stride))
    if max_windows is not None:
        centers = centers[:max_windows]

    raw_features, cp_features, targets = [], [], []
    for center in centers:
        start = center - half
        end = center + half
        packet_start = start * 10
        packet_end = end * 10
        delta, _ = remove_static_component(h[:, :, packet_start:packet_end])
        x, _, _ = build_link_doppler_time_tensor(delta, nperseg=64, noverlap=48, nfft=64)
        x = normalize_tensor(x)
        raw_features.append(x.ravel())
        cp_features.append(cp_feature(x))
        targets.append(gt[center].ravel())
    return np.vstack(raw_features), np.vstack(cp_features), np.vstack(targets)


def evaluate(name, y_true, y_pred):
    return {
        "name": name,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
    }


def main():
    data_root = ROOT / "data" / "MMFi_full" / "extracted" / "E01" / "E01" / "S01"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    train_actions = ["A01", "A02", "A03", "A04"]
    test_actions = ["A05"]
    max_windows_per_action = 48

    train_raw, train_cp, train_y = [], [], []
    test_raw, test_cp, test_y = [], [], []

    for action in train_actions:
        print(f"processing train {action}")
        xr, xc, y = sequence_windows(data_root / action, max_windows=max_windows_per_action)
        train_raw.append(xr)
        train_cp.append(xc)
        train_y.append(y)

    for action in test_actions:
        print(f"processing test {action}")
        xr, xc, y = sequence_windows(data_root / action, max_windows=max_windows_per_action)
        test_raw.append(xr)
        test_cp.append(xc)
        test_y.append(y)

    x_raw_train = np.vstack(train_raw)
    x_cp_train = np.vstack(train_cp)
    y_train = np.vstack(train_y)
    x_raw_test = np.vstack(test_raw)
    x_cp_test = np.vstack(test_cp)
    y_test = np.vstack(test_y)

    alphas = np.logspace(-3, 3, 13)
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)

    raw_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    cp_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    raw_model.fit(x_raw_train, y_train)
    cp_model.fit(x_cp_train, y_train)

    results = [
        evaluate("mean_pose", y_test, mean_pred),
        evaluate("raw_ldt_ridge", y_test, raw_model.predict(x_raw_test)),
        evaluate("cp_rank4_ridge", y_test, cp_model.predict(x_cp_test)),
    ]

    for res in results:
        print(
            f"{res['name']}: MPJPE={res['mpjpe']:.4f} "
            f"PCK_50={res['pck_50']:.3f} "
            f"PCK_40={res['pck_40']:.3f} "
            f"PCK_30={res['pck_30']:.3f} "
            f"PCK_20={res['pck_20']:.3f} "
            f"PCK_10={res['pck_10']:.3f}"
        )

    out_npz = out_dir / "E01_S01_leave_A05_probe.npz"
    np.savez_compressed(
        out_npz,
        x_raw_train=x_raw_train,
        x_cp_train=x_cp_train,
        y_train=y_train,
        x_raw_test=x_raw_test,
        x_cp_test=x_cp_test,
        y_test=y_test,
        **{f"{r['name']}_{k}": v for r in results for k, v in r.items() if k != "name"},
    )

    names = [r["name"] for r in results]
    mpjpes = [r["mpjpe"] for r in results]
    pck20 = [r["pck_20"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(names, mpjpes)
    axes[0].set_ylabel("MPJPE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, pck20)
    axes[1].set_ylabel("HPE-Li PCK_20 (%)")
    axes[1].tick_params(axis="x", rotation=20)
    out_png = out_dir / "E01_S01_leave_A05_probe.png"
    fig.savefig(out_png, dpi=180)
    print(f"train_samples={len(y_train)} test_samples={len(y_test)}")
    print(f"saved_npz={out_npz}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
