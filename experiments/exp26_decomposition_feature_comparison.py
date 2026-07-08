"""Compare PCA, matrix-NMF, Tucker, and CP features for MM-Fi HPE.

This experiment uses the same MM-Fi protocol-3 frame split and the same Ridge
regressor for every feature family. It is intended to answer:

  Does CP provide a useful representation compared with other decomposition
  choices under a matched downstream regressor?

The default subset is chosen to produce a same-day result. Increase
--max-train/--max-test for the paper-scale run.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.decomposition import MiniBatchNMF
except ImportError:  # pragma: no cover
    from sklearn.decomposition import NMF as MiniBatchNMF

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "experiments"))

from exp06_protocol3_frame_cp_probe import (  # noqa: E402
    hpe_li_protocol3_subject_split,
    iter_available_frames,
    load_hpe_li_frame,
    split_val_test,
)
from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


def evaluate(name, y_true, y_pred, train_sec, feature_sec, fit_sec, predict_sec, dim):
    return {
        "name": name,
        "feature_dim": dim,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
        "train_samples": len(y_true),  # overwritten below for readability
        "feature_sec": feature_sec,
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "total_train_sec": train_sec,
    }


def load_raw(items, max_items=None, progress_every=2000):
    if max_items is not None:
        items = items[:max_items]
    frames = []
    y = []
    t0 = time.time()
    for i, item in enumerate(items, start=1):
        frames.append(load_hpe_li_frame(item["frame_path"]).astype(np.float32))
        y.append(item["pose"].ravel().astype(np.float32))
        if i % progress_every == 0:
            print(f"loaded_raw={i}/{len(items)} elapsed={time.time() - t0:.1f}s", flush=True)
    return np.stack(frames), np.vstack(y)


def tucker_frame_feature(x, ranks):
    r0, r1, r2 = ranks
    u0 = np.linalg.svd(np.reshape(np.moveaxis(x, 0, 0), (x.shape[0], -1)), full_matrices=False)[0][:, :r0]
    u1 = np.linalg.svd(np.reshape(np.moveaxis(x, 1, 0), (x.shape[1], -1)), full_matrices=False)[0][:, :r1]
    u2 = np.linalg.svd(np.reshape(np.moveaxis(x, 2, 0), (x.shape[2], -1)), full_matrices=False)[0][:, :r2]
    core = np.einsum("ia,jb,kc,ijk->abc", u0, u1, u2, x)
    return np.concatenate([core.ravel(), u0.ravel(), u1.ravel(), u2.ravel()]).astype(np.float32)


def tucker_features(frames, ranks, progress_every=2000):
    feats = []
    t0 = time.time()
    for i, frame in enumerate(frames, start=1):
        feats.append(tucker_frame_feature(frame, ranks))
        if i % progress_every == 0:
            print(f"tucker_features={i}/{len(frames)} elapsed={time.time() - t0:.1f}s", flush=True)
    return np.vstack(feats)


def fit_predict_ridge(x_train, y_train, x_test, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    t0 = time.time()
    model.fit(x_train, y_train)
    fit_sec = time.time() - t0
    t1 = time.time()
    pred = model.predict(x_test)
    pred_sec = time.time() - t1
    return pred, fit_sec, pred_sec


def run(args):
    train_form, val_form = hpe_li_protocol3_subject_split()
    train_items = list(iter_available_frames(args.data_root, train_form))
    val_items = list(iter_available_frames(args.data_root, val_form))
    _, test_items = split_val_test(val_items)
    print(f"available_train={len(train_items)} available_test={len(test_items)}")
    print(f"using_train={args.max_train} using_test={args.max_test}")

    t_load = time.time()
    train_frames, y_train = load_raw(train_items, args.max_train)
    test_frames, y_test = load_raw(test_items, args.max_test)
    raw_load_sec = time.time() - t_load
    flat_train = train_frames.reshape((len(train_frames), -1))
    flat_test = test_frames.reshape((len(test_frames), -1))

    rows = []
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    rows.append(evaluate("mean_pose", y_test, mean_pred, 0, 0, 0, 0, 0))

    # PCA over flattened CSI frames.
    t0 = time.time()
    pca = PCA(n_components=args.components, svd_solver="randomized", random_state=args.seed)
    x_pca_train = pca.fit_transform(flat_train)
    x_pca_test = pca.transform(flat_test)
    feat_sec = time.time() - t0
    pred, fit_sec, pred_sec = fit_predict_ridge(x_pca_train, y_train, x_pca_test, args.ridge_alpha)
    rows.append(evaluate(f"pca_{args.components}", y_test, pred, fit_sec, feat_sec, fit_sec, pred_sec, x_pca_train.shape[1]))

    # Matrix NMF over flattened nonnegative CSI frames.
    t0 = time.time()
    nmf = MiniBatchNMF(
        n_components=args.components,
        init="nndsvda",
        random_state=args.seed,
        max_iter=args.nmf_iter,
        batch_size=args.nmf_batch_size,
    )
    x_nmf_train = nmf.fit_transform(np.maximum(flat_train, 0))
    x_nmf_test = nmf.transform(np.maximum(flat_test, 0))
    feat_sec = time.time() - t0
    pred, fit_sec, pred_sec = fit_predict_ridge(x_nmf_train, y_train, x_nmf_test, args.ridge_alpha)
    rows.append(evaluate(f"matrix_nmf_{args.components}", y_test, pred, fit_sec, feat_sec, fit_sec, pred_sec, x_nmf_train.shape[1]))

    # Tucker HOSVD features per frame.
    ranks = tuple(int(v) for v in args.tucker_ranks.split(","))
    t0 = time.time()
    x_tucker_train = tucker_features(train_frames, ranks)
    x_tucker_test = tucker_features(test_frames, ranks)
    feat_sec = time.time() - t0
    pred, fit_sec, pred_sec = fit_predict_ridge(x_tucker_train, y_train, x_tucker_test, args.ridge_alpha)
    rows.append(evaluate(f"tucker_{args.tucker_ranks}", y_test, pred, fit_sec, feat_sec, fit_sec, pred_sec, x_tucker_train.shape[1]))

    # CP features from the existing cache when the same prefix subset is used.
    z = np.load(args.cp_cache)
    x_cp_train = z["x_cp_train"][: len(y_train)]
    x_cp_test = z["x_cp_test"][: len(y_test)]
    pred, fit_sec, pred_sec = fit_predict_ridge(x_cp_train, y_train, x_cp_test, args.ridge_alpha)
    rows.append(evaluate("cp_rank4_cached", y_test, pred, fit_sec, 0.0, fit_sec, pred_sec, x_cp_train.shape[1]))

    for row in rows:
        row["train_samples"] = len(y_train)
        row["test_samples"] = len(y_test)
        row["raw_load_sec"] = raw_load_sec
        print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MMFi_full" / "extracted")
    parser.add_argument("--cp-cache", type=Path, default=ROOT / "outputs" / "protocol3_frame_cp_probe_full_cp10.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "decomposition_feature_comparison.csv")
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--tucker-ranks", default="3,4,4")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--nmf-iter", type=int, default=120)
    parser.add_argument("--nmf-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
