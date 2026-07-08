"""Fairer decomposition/regressor ablation.

The earlier exp26 only compared PCA/NMF/Tucker/CP with Ridge. That answers a
linear-probe question, but not the stronger reviewer question:

  Is CP better than other decompositions when each gets a reasonable regressor?

This script compares:
  - PCA + MLP
  - Matrix-NMF + MLP
  - Tucker + MLP
  - CP + MLP
  - CP + S-AFF

All methods use the same MM-Fi protocol subset and predict the same 17x3 pose.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

try:
    from sklearn.decomposition import MiniBatchNMF
except ImportError:  # pragma: no cover
    from sklearn.decomposition import NMF as MiniBatchNMF

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "experiments"))

from exp06_protocol3_frame_cp_probe import (  # noqa: E402
    cp_frame_feature,
    hpe_li_protocol3_subject_split,
    iter_available_frames,
    load_hpe_li_frame,
    split_val_test,
)
from exp14_cp_cnn_aff import cp_as_component_image, standardize, standardize_y  # noqa: E402
from exp24_hard_routed_saff import make_routable_saff  # noqa: E402
from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


def load_raw(items, max_items=None, progress_every=2000):
    if max_items is not None:
        items = items[:max_items]
    frames, y = [], []
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


def cp_features_from_frames(frames, rank, iters, progress_every=2000):
    feats = []
    t0 = time.time()
    for i, frame in enumerate(frames, start=1):
        feats.append(cp_frame_feature(frame, rank=rank, iters=iters))
        if i % progress_every == 0:
            print(f"cp_features={i}/{len(frames)} elapsed={time.time() - t0:.1f}s", flush=True)
    return np.vstack(feats)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def standardize_xy(x_train, x_test, eps=1e-6):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    return (x_train - mean) / (std + eps), (x_test - mean) / (std + eps)


def train_predict_mlp(x_train, y_train_s, x_test, args):
    model = MLP(x_train.shape[1], y_train_s.shape[1], args.hidden, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train_s.astype(np.float32)))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"mlp epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    preds = []
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size].astype(np.float32)).to(args.device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds), train_sec, time.time() - t1, sum(p.numel() for p in model.parameters())


def train_predict_saff(x_train_cp, y_train_s, x_test_cp, args):
    _, nn_mod, DataLoader, TensorDataset = __import__("exp14_cp_cnn_aff").require_torch()
    model = make_routable_saff(nn_mod, args.gate_temperature).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train_cp.astype(np.float32)), torch.from_numpy(y_train_s.astype(np.float32)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            pred, gates = model(xb, return_gates=True)
            loss = loss_fn(pred, yb)
            if args.gate_entropy_weight > 0:
                from exp14_cp_cnn_aff import gate_entropy

                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"saff epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    preds = []
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_test_cp), args.batch_size):
            xb = torch.from_numpy(x_test_cp[start : start + args.batch_size].astype(np.float32)).to(args.device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds), train_sec, time.time() - t1, sum(p.numel() for p in model.parameters())


def evaluate(name, y_true, y_pred, feature_dim, regressor, params, train_sec, predict_sec):
    return {
        "name": name,
        "regressor": regressor,
        "feature_dim": feature_dim,
        "params": params,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
        "train_sec": train_sec,
        "predict_sec": predict_sec,
    }


def run(args):
    train_form, val_form = hpe_li_protocol3_subject_split()
    train_items = list(iter_available_frames(args.data_root, train_form))
    val_items = list(iter_available_frames(args.data_root, val_form))
    _, test_items = split_val_test(val_items)
    print(f"available_train={len(train_items)} available_test={len(test_items)}")

    train_frames, y_train = load_raw(train_items, args.max_train)
    test_frames, y_test = load_raw(test_items, args.max_test)
    flat_train = train_frames.reshape((len(train_frames), -1))
    flat_test = test_frames.reshape((len(test_frames), -1))
    y_train_s, _, y_mean, y_std = standardize_y(y_train, y_test)

    rows = []
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    rows.append(evaluate("mean_pose", y_test, mean_pred, 0, "none", 0, 0, 0))

    feature_sets = []
    t0 = time.time()
    pca = PCA(n_components=args.components, svd_solver="randomized", random_state=args.seed)
    feature_sets.append(("pca", *standardize_xy(pca.fit_transform(flat_train), pca.transform(flat_test))))
    print(f"built_pca elapsed={time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    nmf = MiniBatchNMF(
        n_components=args.components,
        init="nndsvda",
        random_state=args.seed,
        max_iter=args.nmf_iter,
        batch_size=args.nmf_batch_size,
    )
    feature_sets.append(("matrix_nmf", *standardize_xy(nmf.fit_transform(np.maximum(flat_train, 0)), nmf.transform(np.maximum(flat_test, 0)))))
    print(f"built_nmf elapsed={time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    ranks = tuple(int(v) for v in args.tucker_ranks.split(","))
    feature_sets.append(("tucker", *standardize_xy(tucker_features(train_frames, ranks), tucker_features(test_frames, ranks))))
    print(f"built_tucker elapsed={time.time() - t0:.1f}s", flush=True)

    x_cp_train_vec = None
    x_cp_test_vec = None
    if args.cp_source in ("auto", "cache") and args.cp_cache.exists():
        z = np.load(args.cp_cache)
        cache_ok = (
            "x_cp_train" in z
            and "x_cp_test" in z
            and len(z["x_cp_train"]) == len(y_train)
            and len(z["x_cp_test"]) == len(y_test)
        )
        if cache_ok:
            print(f"using_cp_cache={args.cp_cache}", flush=True)
            x_cp_train_vec = z["x_cp_train"].astype(np.float32)
            x_cp_test_vec = z["x_cp_test"].astype(np.float32)
        else:
            msg = (
                f"cp_cache_mismatch cache_train={len(z['x_cp_train']) if 'x_cp_train' in z else 'NA'} "
                f"cache_test={len(z['x_cp_test']) if 'x_cp_test' in z else 'NA'} "
                f"current_train={len(y_train)} current_test={len(y_test)}"
            )
            if args.cp_source == "cache":
                raise ValueError(msg)
            print(msg, flush=True)
            print("computing_cp_from_current_raw_frames", flush=True)
    if x_cp_train_vec is None or x_cp_test_vec is None:
        if args.cp_source == "cache":
            raise ValueError(f"CP cache unavailable or incompatible: {args.cp_cache}")
        x_cp_train_vec = cp_features_from_frames(train_frames, args.cp_rank, args.cp_iters)
        x_cp_test_vec = cp_features_from_frames(test_frames, args.cp_rank, args.cp_iters)
        if args.computed_cp_cache is not None:
            args.computed_cp_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.computed_cp_cache,
                x_cp_train=x_cp_train_vec,
                x_cp_test=x_cp_test_vec,
                y_train=y_train,
                y_test=y_test,
                cp_rank=args.cp_rank,
                cp_iters=args.cp_iters,
            )
            print(f"saved_computed_cp_cache={args.computed_cp_cache}", flush=True)
    feature_sets.append(("cp", *standardize_xy(x_cp_train_vec, x_cp_test_vec)))

    for name, xtr, xte in feature_sets:
        print(f"training {name}+MLP", flush=True)
        pred_s, train_sec, pred_sec, params = train_predict_mlp(xtr, y_train_s, xte, args)
        pred = pred_s * y_std + y_mean
        rows.append(evaluate(f"{name}_mlp", y_test, pred, xtr.shape[1], "MLP", params, train_sec, pred_sec))
        print(rows[-1], flush=True)

    print("training cp+S-AFF", flush=True)
    x_cp_train_img = cp_as_component_image(x_cp_train_vec)
    x_cp_test_img = cp_as_component_image(x_cp_test_vec)
    x_cp_train_img, x_cp_test_img = standardize(x_cp_train_img, x_cp_test_img)
    pred_s, train_sec, pred_sec, params = train_predict_saff(x_cp_train_img, y_train_s, x_cp_test_img, args)
    pred = pred_s * y_std + y_mean
    rows.append(evaluate("cp_saff", y_test, pred, x_cp_train_vec.shape[1], "S-AFF", params, train_sec, pred_sec))
    print(rows[-1], flush=True)

    for row in rows:
        row["train_samples"] = len(y_train)
        row["test_samples"] = len(y_test)

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
    parser.add_argument("--cp-source", choices=["auto", "cache", "compute"], default="auto")
    parser.add_argument("--computed-cp-cache", type=Path, default=None)
    parser.add_argument("--cp-rank", type=int, default=4)
    parser.add_argument("--cp-iters", type=int, default=35)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "decomposition_regressor_ablation.csv")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--tucker-ranks", default="3,4,4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nmf-iter", type=int, default=80)
    parser.add_argument("--nmf-batch-size", type=int, default=1024)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
