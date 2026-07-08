from pathlib import Path
import argparse
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


RANK = 4
LINKS = 3
SUBCARRIERS = 114
PACKETS = 10
A_SIZE = LINKS * RANK
B_SIZE = SUBCARRIERS * RANK
C_SIZE = PACKETS * RANK


def component_indices(r):
    a = np.array([l * RANK + r for l in range(LINKS)])
    b0 = A_SIZE
    b = b0 + np.array([s * RANK + r for s in range(SUBCARRIERS)])
    c0 = A_SIZE + B_SIZE
    c = c0 + np.array([p * RANK + r for p in range(PACKETS)])
    return np.concatenate([a, b, c])


def mode_indices(mode):
    if mode == "link_A":
        return np.arange(0, A_SIZE)
    if mode == "subcarrier_B":
        return np.arange(A_SIZE, A_SIZE + B_SIZE)
    if mode == "packet_C":
        return np.arange(A_SIZE + B_SIZE, A_SIZE + B_SIZE + C_SIZE)
    raise ValueError(mode)


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


def print_result(res, base=None):
    suffix = ""
    if base is not None:
        suffix = f" dPCK20={res['pck_20'] - base['pck_20']:.3f}"
    print(
        f"{res['name']}: MPJPE={res['mpjpe']:.3f} "
        f"PCK_50={res['pck_50']:.3f} "
        f"PCK_40={res['pck_40']:.3f} "
        f"PCK_30={res['pck_30']:.3f} "
        f"PCK_20={res['pck_20']:.3f} "
        f"PCK_10={res['pck_10']:.3f}{suffix}"
    )


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def aggregate_importance(importances):
    rows = []
    for r in range(RANK):
        idx = component_indices(r)
        rows.append(
            {
                "group": f"component_{r}",
                "importance": float(np.sum(importances[idx])),
            }
        )
    for mode in ["link_A", "subcarrier_B", "packet_C"]:
        idx = mode_indices(mode)
        rows.append({"group": mode, "importance": float(np.sum(importances[idx]))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "protocol3_frame_cp_probe_available.npz",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=ROOT / "outputs" / "component_pose_relevance_ablation.csv",
    )
    parser.add_argument(
        "--importance-output",
        type=Path,
        default=ROOT / "outputs" / "component_pose_relevance_importance.csv",
    )
    args = parser.parse_args()

    data_path = args.input
    out_ablation = args.ablation_output
    out_importance = args.importance_output

    z = np.load(data_path)
    x_train = z["x_cp_train"]
    y_train = z["y_train"]
    x_test = z["x_cp_test"]
    y_test = z["y_test"]

    print(f"train={x_train.shape} test={x_test.shape}")
    model = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=4,
        random_state=0,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(x_train, y_train)
    print(f"fit_sec={time.time() - t0:.1f}")

    rows = []
    pred = model.predict(x_test)
    base = evaluate("all_components", y_test, pred)
    rows.append(base)
    print_result(base)

    for r in range(RANK):
        x_drop = x_test.copy()
        x_drop[:, component_indices(r)] = 0.0
        res = evaluate(f"drop_component_{r}", y_test, model.predict(x_drop))
        rows.append(res)
        print_result(res, base=base)

    for r in range(RANK):
        x_only = np.zeros_like(x_test)
        idx = component_indices(r)
        x_only[:, idx] = x_test[:, idx]
        res = evaluate(f"only_component_{r}", y_test, model.predict(x_only))
        rows.append(res)
        print_result(res, base=base)

    for mode in ["link_A", "subcarrier_B", "packet_C"]:
        x_drop = x_test.copy()
        x_drop[:, mode_indices(mode)] = 0.0
        res = evaluate(f"drop_{mode}", y_test, model.predict(x_drop))
        rows.append(res)
        print_result(res, base=base)

    write_csv(out_ablation, rows)
    importance_rows = aggregate_importance(model.feature_importances_)
    write_csv(out_importance, importance_rows)
    print(f"saved_ablation={out_ablation}")
    print(f"saved_importance={out_importance}")


if __name__ == "__main__":
    main()
