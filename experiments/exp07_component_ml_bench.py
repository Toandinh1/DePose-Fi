from pathlib import Path
import argparse
import sys
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


def evaluate(name, y_true, y_pred, train_time, predict_time, params=None):
    return {
        "name": name,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
        "train_sec": train_time,
        "predict_sec": predict_time,
        "us_per_sample": 1e6 * predict_time / len(y_true),
        "params": params if params is not None else np.nan,
    }


def fit_predict(name, model, x_train, y_train, x_test, y_test, params=None):
    print(f"training {name}", flush=True)
    t0 = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - t0
    t1 = time.time()
    pred = model.predict(x_test)
    predict_time = time.time() - t1
    res = evaluate(name, y_test, pred, train_time, predict_time, params=params)
    print_result(res)
    return res


def print_result(res):
    print(
        f"{res['name']}: MPJPE={res['mpjpe']:.3f} "
        f"PCK_50={res['pck_50']:.3f} "
        f"PCK_40={res['pck_40']:.3f} "
        f"PCK_30={res['pck_30']:.3f} "
        f"PCK_20={res['pck_20']:.3f} "
        f"PCK_10={res['pck_10']:.3f} "
        f"us/sample={res['us_per_sample']:.2f}"
    )


def linear_param_count(in_dim, out_dim):
    return in_dim * out_dim + out_dim


def mlp_param_count(in_dim, hidden, out_dim):
    return in_dim * hidden + hidden + hidden * out_dim + out_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "protocol3_frame_cp_probe_available.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "component_ml_bench_available.csv",
    )
    args = parser.parse_args()

    data_path = args.input
    out_csv = args.output
    z = np.load(data_path)
    x_cp_train = z["x_cp_train"]
    x_cp_test = z["x_cp_test"]
    x_stats_train = z["x_stats_train"]
    x_stats_test = z["x_stats_test"]
    y_train = z["y_train"]
    y_test = z["y_test"]

    results = []
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    results.append(evaluate("mean_pose", y_test, mean_pred, 0.0, 0.0, params=51))
    print_result(results[-1])

    results.append(
        fit_predict(
            "raw_stats_ridge",
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            x_stats_train,
            y_train,
            x_stats_test,
            y_test,
            params=linear_param_count(x_stats_train.shape[1], y_train.shape[1]),
        )
    )
    results.append(
        fit_predict(
            "raw_stats_mlp64",
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
                    verbose=False,
                ),
            ),
            x_stats_train,
            y_train,
            x_stats_test,
            y_test,
            params=mlp_param_count(x_stats_train.shape[1], 64, y_train.shape[1]),
        )
    )
    results.append(
        fit_predict(
            "raw_stats_extratrees",
            ExtraTreesRegressor(
                n_estimators=80,
                max_depth=24,
                min_samples_leaf=4,
                random_state=0,
                n_jobs=-1,
            ),
            x_stats_train,
            y_train,
            x_stats_test,
            y_test,
        )
    )
    results.append(
        fit_predict(
            "cp_components_ridge",
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            params=linear_param_count(x_cp_train.shape[1], y_train.shape[1]),
        )
    )
    results.append(
        fit_predict(
            "cp_components_pls16",
            make_pipeline(StandardScaler(), PLSRegression(n_components=16, scale=False)),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            params=x_cp_train.shape[1] * 16 + 16 * y_train.shape[1],
        )
    )
    results.append(
        fit_predict(
            "cp_components_mlp64",
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
                    verbose=False,
                ),
            ),
            x_cp_train,
            y_train,
            x_cp_test,
            y_test,
            params=mlp_param_count(x_cp_train.shape[1], 64, y_train.shape[1]),
        )
    )
    results.append(
        fit_predict(
            "cp_components_extratrees",
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
        )
    )

    header = list(results[0].keys())
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for res in results:
            f.write(",".join(str(res[k]) for k in header) + "\n")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
