import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import nonnegative_cp_mu  # noqa: E402
from metrics import mpjpe_3d  # noqa: E402


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for AFF. Install torch and rerun.") from exc
    return torch, nn, DataLoader, TensorDataset


def pck_3d(y_true, y_pred, threshold, torso_pair=(1, 11), eps=1e-8):
    y_true = np.asarray(y_true).reshape((len(y_true), -1, 3))
    y_pred = np.asarray(y_pred).reshape((len(y_pred), -1, 3))
    i, j = torso_pair
    if y_true.shape[1] <= max(i, j):
        scale = np.ones((len(y_true),), dtype=np.float32)
    else:
        scale = np.linalg.norm(y_true[:, i, :] - y_true[:, j, :], axis=1)
    scale = np.maximum(scale, eps)
    dist = np.linalg.norm(y_pred - y_true, axis=2) / scale[:, None]
    return float(100.0 * np.mean(dist <= threshold))


def normalize_frame(frame):
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    if np.iscomplexobj(frame):
        frame = np.abs(frame).astype(np.float32)
    mn = float(frame.min())
    mx = float(frame.max())
    if mx > mn:
        return (frame - mn) / (mx - mn)
    return np.zeros_like(frame, dtype=np.float32)


def cp_feature(frame, rank, iters):
    a, b, c, _ = nonnegative_cp_mu(normalize_frame(frame), rank=rank, iters=iters, seed=rank)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()]).astype(np.float32)


def raw_stats_feature(frame):
    frame = normalize_frame(frame)
    return np.concatenate(
        [
            frame.mean(axis=(1, 2)),
            frame.std(axis=(1, 2)),
            frame.mean(axis=(0, 2)),
            frame.std(axis=(0, 2)),
            frame.mean(axis=(0, 1)),
            frame.std(axis=(0, 1)),
        ]
    ).astype(np.float32)


def split_train_test(n, train_ratio=0.7, seed=0):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    split = int(np.floor(train_ratio * n))
    return order[:split], order[split:]


def canonicalize_pose(y, person_policy="first"):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 4:
        if person_policy != "first":
            raise ValueError("Only --person-policy first is implemented for multi-person pose arrays.")
        y = y[:, 0]
    if y.ndim != 3 or y.shape[-1] != 3:
        raise ValueError(f"Expected pose shape N x J x 3 or N x M x J x 3, got {y.shape}")
    return y


def featurize(x, rank, iters, progress_every=1000):
    x_cp = []
    x_stats = []
    t0 = time.time()
    for idx, frame in enumerate(x, start=1):
        x_cp.append(cp_feature(frame, rank=rank, iters=iters))
        x_stats.append(raw_stats_feature(frame))
        if idx % progress_every == 0:
            print(f"featurized={idx}/{len(x)} elapsed_sec={time.time() - t0:.1f}", flush=True)
    return np.vstack(x_cp), np.vstack(x_stats)


def standardize(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps)


def standardize_y(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps), mean, std


def make_aff(nn, rank, links, subcarriers, packets, out_dim):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.links = links
            self.subcarriers = subcarriers
            self.packets = packets
            self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(rank * links, 32), nn.ReLU())
            self.b_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(rank, rank),
                nn.Sigmoid(),
            )
            self.b_net = nn.Sequential(
                nn.Conv1d(rank, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(32 * 8, 96),
                nn.ReLU(),
            )
            self.c_net = nn.Sequential(
                nn.Conv1d(rank, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
                nn.Linear(16 * 4, 32),
                nn.ReLU(),
            )
            self.gate = nn.Sequential(nn.Linear(160, 3), nn.Softmax(dim=1))
            self.heads = nn.ModuleList(
                [nn.Linear(32, out_dim), nn.Linear(96, out_dim), nn.Linear(32, out_dim)]
            )

        def forward(self, x):
            a = x[:, : rank * links].reshape((-1, rank, links))
            b0 = rank * links
            b1 = b0 + rank * subcarriers
            b = x[:, b0:b1].reshape((-1, rank, subcarriers))
            c = x[:, b1:].reshape((-1, rank, packets))
            fa = self.a_net(a)
            fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
            fc = self.c_net(c)
            gates = self.gate(nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1))
            return (
                gates[:, 0:1] * self.heads[0](fa)
                + gates[:, 1:2] * self.heads[1](fb)
                + gates[:, 2:3] * self.heads[2](fc)
            )

    return Net()


def train_aff(x_train, y_train, x_test, args, dims, out_dim):
    torch, nn, DataLoader, TensorDataset = require_torch()
    device = torch.device(args.device)
    model = make_aff(nn, args.rank, *dims, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    preds = []
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_sec = time.time() - t1
    params = sum(p.numel() for p in model.parameters())
    return np.vstack(preds), train_sec, pred_sec, params


def evaluate(name, y_true, y_pred, train_sec=0.0, pred_sec=0.0, params=0):
    return {
        "name": name,
        "mpjpe": mpjpe_3d(y_true, y_pred, num_joints=y_true.reshape((len(y_true), -1, 3)).shape[1]),
        "pck_50": pck_3d(y_true, y_pred, 0.5),
        "pck_40": pck_3d(y_true, y_pred, 0.4),
        "pck_30": pck_3d(y_true, y_pred, 0.3),
        "pck_20": pck_3d(y_true, y_pred, 0.2),
        "pck_10": pck_3d(y_true, y_pred, 0.1),
        "train_sec": train_sec,
        "predict_sec": pred_sec,
        "us_per_sample": 0.0 if len(y_true) == 0 else 1e6 * pred_sec / len(y_true),
        "params": params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Canonical NPZ with arrays `wifi` and `pose`.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "canonical_cp_aff.csv")
    parser.add_argument("--features-output", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--person-policy", choices=["first"], default="first")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    z = np.load(args.input, allow_pickle=False)
    if "wifi" not in z or "pose" not in z:
        raise SystemExit("Canonical NPZ must contain arrays named `wifi` and `pose`.")
    wifi = np.asarray(z["wifi"])
    pose = canonicalize_pose(z["pose"], args.person_policy)
    if wifi.ndim != 4:
        raise ValueError(f"Expected wifi shape N x L x S x P, got {wifi.shape}")
    if len(wifi) != len(pose):
        raise ValueError(f"wifi/pose length mismatch: {len(wifi)} vs {len(pose)}")
    if args.max_samples is not None:
        wifi = wifi[: args.max_samples]
        pose = pose[: args.max_samples]

    train_idx, test_idx = split_train_test(len(wifi), args.train_ratio, args.seed)
    x_cp, x_stats = featurize(wifi, args.rank, args.cp_iters)
    y = pose.reshape((len(pose), -1)).astype(np.float32)

    x_cp_train, x_cp_test = x_cp[train_idx], x_cp[test_idx]
    x_stats_train, x_stats_test = x_stats[train_idx], x_stats[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rows = []
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    rows.append(evaluate("mean_pose", y_test, mean_pred))

    stats_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    cp_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    t0 = time.time()
    stats_model.fit(x_stats_train, y_train)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = stats_model.predict(x_stats_test)
    rows.append(evaluate("raw_stats_ridge", y_test, pred, train_sec, time.time() - t1, 0))

    t0 = time.time()
    cp_model.fit(x_cp_train, y_train)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = cp_model.predict(x_cp_test)
    rows.append(evaluate("cp_ridge", y_test, pred, train_sec, time.time() - t1, x_cp_train.shape[1] * y_train.shape[1]))

    x_aff_train, x_aff_test = standardize(x_cp_train, x_cp_test)
    y_train_s, _, y_mean, y_std = standardize_y(y_train, y_test)
    pred_s, train_sec, pred_sec, params = train_aff(
        x_aff_train,
        y_train_s,
        x_aff_test,
        args,
        dims=wifi.shape[1:],
        out_dim=y_train.shape[1],
    )
    rows.append(evaluate("cp_aff", y_test, pred_s * y_std + y_mean, train_sec, pred_sec, params))

    args.output.parent.mkdir(exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.features_output:
        np.savez_compressed(
            args.features_output,
            x_cp_train=x_cp_train,
            x_cp_test=x_cp_test,
            x_stats_train=x_stats_train,
            x_stats_test=x_stats_test,
            y_train=y_train,
            y_test=y_test,
        )
    for row in rows:
        print(row)
    print(f"saved_csv={args.output}")


if __name__ == "__main__":
    main()
