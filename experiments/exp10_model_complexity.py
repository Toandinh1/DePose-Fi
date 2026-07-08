import pickle
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


def model_bytes(model):
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def linear_params(in_dim, out_dim):
    return in_dim * out_dim + out_dim


def mlp_params(in_dim, hidden, out_dim):
    return in_dim * hidden + hidden + hidden * out_dim + out_dim


def extratrees_stats(model):
    nodes = sum(est.tree_.node_count for est in model.estimators_)
    leaves = sum(est.tree_.n_leaves for est in model.estimators_)
    return nodes, leaves


def evaluate(y_true, y_pred):
    return {
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
    }


def bench(name, model, x_train, y_train, x_test, y_test, params):
    print(f"training {name}", flush=True)
    t0 = time.time()
    model.fit(x_train, y_train)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = model.predict(x_test)
    pred_sec = time.time() - t1
    metrics = evaluate(y_test, pred)
    size = model_bytes(model)
    nodes = np.nan
    leaves = np.nan
    if isinstance(model, ExtraTreesRegressor):
        nodes, leaves = extratrees_stats(model)
    row = {
        "name": name,
        "params": params,
        "model_size_mb": size / (1024 * 1024),
        "tree_nodes": nodes,
        "tree_leaves": leaves,
        "train_sec": train_sec,
        "predict_sec": pred_sec,
        "us_per_sample": 1e6 * pred_sec / len(y_test),
        **metrics,
    }
    print(row)
    return row


def write_csv(path, rows):
    keys = list(rows[0])
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def main():
    z = np.load(ROOT / "outputs" / "protocol3_frame_cp_probe_available.npz")
    x_cp_train = z["x_cp_train"]
    x_cp_test = z["x_cp_test"]
    y_train = z["y_train"]
    y_test = z["y_test"]

    rows = []
    rows.append(
        bench(
            "cp_ridge",
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            linear_params(x_cp_train.shape[1], y_train.shape[1]),
        )
    )
    rows.append(
        bench(
            "cp_mlp64",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    n_iter_no_change=8,
                    random_state=0,
                ),
            ),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            mlp_params(x_cp_train.shape[1], 64, y_train.shape[1]),
        )
    )
    rows.append(
        bench(
            "cp_extratrees",
            ExtraTreesRegressor(
                n_estimators=80,
                max_depth=24,
                min_samples_leaf=4,
                random_state=0,
                n_jobs=-1,
            ),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            np.nan,
        )
    )
    out = ROOT / "outputs" / "model_complexity_available.csv"
    write_csv(out, rows)
    print(f"saved_csv={out}")


if __name__ == "__main__":
    main()
