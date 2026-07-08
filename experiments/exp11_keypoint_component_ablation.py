from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import hpe_li_pck_mmfi  # noqa: E402


RANK = 4
LINKS = 3
SUBCARRIERS = 114
PACKETS = 10
A_SIZE = LINKS * RANK
B_SIZE = SUBCARRIERS * RANK
C_SIZE = PACKETS * RANK
JOINTS = [f"J{i:02d}" for i in range(17)]


def component_indices(r):
    a = np.array([l * RANK + r for l in range(LINKS)])
    b = A_SIZE + np.array([s * RANK + r for s in range(SUBCARRIERS)])
    c = A_SIZE + B_SIZE + np.array([p * RANK + r for p in range(PACKETS)])
    return np.concatenate([a, b, c])


def keypoint_pck20(y_true, y_pred, eps=1e-8):
    gt = y_true.reshape((-1, 17, 3))[:, :, :2]
    pred = y_pred.reshape((-1, 17, 3))[:, :, :2]
    scale = np.linalg.norm(gt[:, 1, :] - gt[:, 11, :], axis=1)
    scale = np.maximum(scale, eps)
    dist = np.linalg.norm(pred - gt, axis=2) / scale[:, None]
    return 100.0 * np.mean(dist <= 0.2, axis=0)


def write_csv(path, header, rows):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def main():
    z = np.load(ROOT / "outputs" / "protocol3_frame_cp_probe_available.npz")
    x_train = z["x_cp_train"]
    y_train = z["y_train"]
    x_test = z["x_cp_test"]
    y_test = z["y_test"]

    model = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=4,
        random_state=0,
        n_jobs=-1,
    )
    print(f"training ExtraTrees on {x_train.shape}")
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f"fit_sec={time.time() - t0:.1f}")

    base_pred = model.predict(x_test)
    base_joint = keypoint_pck20(y_test, base_pred)
    print(f"overall_base_pck20={hpe_li_pck_mmfi(y_test, base_pred, 0.2):.3f}")

    drops = []
    rows = []
    for r in range(RANK):
        x_drop = x_test.copy()
        x_drop[:, component_indices(r)] = 0.0
        pred = model.predict(x_drop)
        joint = keypoint_pck20(y_test, pred)
        drop = base_joint - joint
        drops.append(drop)
        for j, name in enumerate(JOINTS):
            rows.append([f"component_{r}", name, base_joint[j], joint[j], drop[j]])
        print(f"component_{r}_mean_joint_drop={drop.mean():.3f} max_drop={drop.max():.3f}")

    out_csv = ROOT / "outputs" / "keypoint_component_ablation.csv"
    write_csv(out_csv, ["component", "joint", "base_pck20", "ablated_pck20", "drop"], rows)

    mat = np.vstack(drops)
    fig, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="Reds")
    ax.set_yticks(np.arange(RANK))
    ax.set_yticklabels([f"C{r}" for r in range(RANK)])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels(JOINTS, rotation=45, ha="right")
    ax.set_ylabel("Dropped component")
    ax.set_title("Keypoint-wise PCK20 drop after component removal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PCK20 drop")
    out_pdf = ROOT / "PAPER" / "figures" / "fig5_keypoint_component_ablation.pdf"
    out_png = ROOT / "PAPER" / "figures" / "fig5_keypoint_component_ablation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    print(f"saved_csv={out_csv}")
    print(f"saved_pdf={out_pdf}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
