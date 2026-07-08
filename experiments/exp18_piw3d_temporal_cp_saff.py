import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "experiments"))

from exp17_piw3d_cp_saff import (  # noqa: E402
    JOINTS,
    LINKS,
    MAX_PEOPLE,
    PACKETS,
    evaluate,
    gate_entropy,
    mean_pose_prediction,
    query_set_loss,
    standardize,
    standardize_pose,
)


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for temporal CP+S-AFF training.") from exc
    return torch, nn, DataLoader, Dataset


def frame_id(name):
    parts = str(name).split("_")
    if len(parts) < 3:
        return str(name), None
    try:
        return "_".join(parts[:-1]), int(parts[-1])
    except ValueError:
        return "_".join(parts[:-1]), None


def temporal_indices(names, radius):
    lookup = {}
    parsed = []
    for idx, name in enumerate(names):
        seq, t = frame_id(name)
        parsed.append((seq, t))
        if t is not None:
            lookup[(seq, t)] = idx
    rows = []
    offsets = list(range(-radius, radius + 1))
    for idx, (seq, t) in enumerate(parsed):
        row = []
        for off in offsets:
            j = lookup.get((seq, t + off)) if t is not None else None
            row.append(idx if j is None else j)
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def filter_people(x, y, mask, names, people):
    if people <= 0:
        return x, y, mask, names
    idx = mask.sum(axis=1).astype(int) == people
    return x[idx], y[idx], mask[idx], names[idx]


def infer_subcarriers(feature_dim, rank):
    modes = feature_dim // rank
    subcarriers = modes - LINKS - PACKETS
    if feature_dim % rank != 0 or subcarriers <= 0:
        raise ValueError(f"Cannot infer CP dimensions from feature_dim={feature_dim}, rank={rank}")
    return subcarriers


def saff_dims(model_size):
    if model_size == "small":
        return 32, 96, 32, 96, 32, 16
    if model_size == "medium":
        return 64, 192, 64, 192, 64, 32
    return 96, 256, 96, 256, 96, 48


def make_temporal_query_saff(nn, rank, feature_dim, model_size, num_queries, query_mixer, temporal_mixer, temperature):
    torch, _, _, _ = require_torch()
    subcarriers = infer_subcarriers(feature_dim, rank)
    a_size = LINKS * rank
    b_size = subcarriers * rank
    a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = saff_dims(model_size)
    hidden = f_dim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
            self.query_mixer = query_mixer
            self.temporal_mixer = temporal_mixer
            self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(rank * LINKS, a_dim), nn.ReLU())
            self.b_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(rank, rank),
                nn.Sigmoid(),
            )
            self.b_net = nn.Sequential(
                nn.Conv1d(rank, b_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(b_channels, b_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(6),
                nn.Flatten(),
                nn.Linear(b_channels * 6, b_dim),
                nn.ReLU(),
            )
            self.c_net = nn.Sequential(
                nn.Conv1d(rank, c_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
                nn.Linear(c_channels * 4, c_dim),
                nn.ReLU(),
            )
            total_dim = a_dim + b_dim + c_dim
            self.fuse_net = nn.Sequential(nn.Linear(total_dim, f_dim), nn.ReLU())
            self.gate = nn.Linear(total_dim, 4)
            self.branch_proj = nn.ModuleList(
                [nn.Linear(a_dim, hidden), nn.Linear(b_dim, hidden), nn.Linear(c_dim, hidden), nn.Linear(f_dim, hidden)]
            )
            if temporal_mixer == "gru":
                self.temporal = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
            elif temporal_mixer == "conv":
                self.temporal = nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            self.query_embed = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
            self.query_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            if query_mixer == "attention":
                heads = 4 if hidden % 4 == 0 else 2
                self.q_mixer = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.q_norm = nn.LayerNorm(hidden)
            self.pose_head = nn.Linear(hidden, JOINTS * 3)
            self.cls_head = nn.Linear(hidden, 1)
            self.count_head = nn.Linear(hidden, MAX_PEOPLE + 1)

        def encode_frame(self, x):
            a = x[:, :a_size].reshape((-1, rank, LINKS))
            b = x[:, a_size : a_size + b_size].reshape((-1, rank, subcarriers))
            c = x[:, a_size + b_size :].reshape((-1, rank, PACKETS))
            fa = self.a_net(a)
            fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
            fc = self.c_net(c)
            h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
            ff = self.fuse_net(h)
            gates = torch.softmax(self.gate(h) / self.temperature, dim=1)
            branch_h = torch.stack(
                [self.branch_proj[0](fa), self.branch_proj[1](fb), self.branch_proj[2](fc), self.branch_proj[3](ff)],
                dim=1,
            )
            base = torch.sum(gates.unsqueeze(-1) * branch_h, dim=1)
            return base, gates

        def forward(self, x_seq, return_gates=False):
            bsz, steps, feat = x_seq.shape
            flat = x_seq.reshape(bsz * steps, feat)
            base, gates = self.encode_frame(flat)
            base = base.reshape(bsz, steps, hidden)
            gates = gates.reshape(bsz, steps, 4)
            if self.temporal_mixer == "gru":
                z, _ = self.temporal(base)
            elif self.temporal_mixer == "conv":
                z = self.temporal(base.transpose(1, 2)).transpose(1, 2)
            else:
                z = base
            center = base[:, steps // 2] if self.temporal_mixer == "none" else base[:, steps // 2] + z[:, steps // 2]
            q = center.unsqueeze(1) + self.query_embed.unsqueeze(0)
            q = self.query_net(q)
            if self.query_mixer == "attention":
                mixed, _ = self.q_mixer(q, q, q, need_weights=False)
                q = self.q_norm(q + mixed)
            poses = self.pose_head(q).reshape((-1, num_queries, JOINTS, 3))
            logits = self.cls_head(q).squeeze(-1)
            count_logits = self.count_head(center)
            center_gates = gates[:, steps // 2]
            if return_gates:
                return poses, logits, count_logits, center_gates
            return poses, logits, count_logits

    return Net()


class WindowDataset:
    def __init__(self, torch, x, y, mask, win_idx):
        self.x = torch.from_numpy(x[win_idx].astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


def train_temporal(x_train, y_train, mask_train, names_train, x_test, args):
    torch, nn, DataLoader, _ = require_torch()
    device = torch.device(args.device)
    train_idx = temporal_indices(names_train, args.temporal_radius)
    model = make_temporal_query_saff(
        nn,
        args.rank,
        x_train.shape[1],
        args.model_size,
        args.num_queries,
        args.query_mixer,
        args.temporal_mixer,
        args.gate_temperature,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = WindowDataset(torch, x_train, y_train, mask_train, train_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            poses, logits, count_logits, gates = model(xb, return_gates=True)
            loss = query_set_loss(
                poses,
                logits,
                count_logits,
                yb,
                mb,
                torch,
                args.cls_weight,
                args.count_weight,
                args.bone_weight,
                args.pose_loss,
            )
            if args.gate_entropy_weight > 0:
                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    return model, train_sec, params


def predict_temporal(model, x, names, args):
    torch, _, _, _ = require_torch()
    device = torch.device(args.device)
    win_idx = temporal_indices(names, args.temporal_radius)
    x_t = torch.from_numpy(x[win_idx].astype(np.float32))
    preds = []
    scores = []
    gates_all = []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb = x_t[start : start + args.batch_size].to(device)
            poses, logits, _, gates = model(xb, return_gates=True)
            preds.append(poses.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    pred_sec = time.time() - t0
    pred = np.vstack(preds)
    score = np.vstack(scores)
    top = np.argsort(-score, axis=1)[:, :MAX_PEOPLE]
    pred_top = np.take_along_axis(pred, top[:, :, None, None], axis=1)
    return pred_top, np.vstack(gates_all), pred_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "piw3d_full_sanitized_phase_rank16_features.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "piw3d_temporal_cp_saff.csv")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--people-filter", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--temporal-mixer", choices=["gru", "conv", "none"], default="gru")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--query-mixer", choices=["none", "attention"], default="attention")
    parser.add_argument("--pose-loss", choices=["mse", "l1", "smooth_l1"], default="l1")
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--bone-weight", type=float, default=0.02)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    z = np.load(args.cache)
    x_train, y_train, mask_train, names_train = filter_people(
        z["x_cp_train"], z["y_train"], z["mask_train"], z["train_names"], args.people_filter
    )
    x_test, y_test, mask_test, names_test = filter_people(
        z["x_cp_test"], z["y_test"], z["mask_test"], z["test_names"], args.people_filter
    )
    print(f"train={len(x_train)} test={len(x_test)} people_filter={args.people_filter}", flush=True)
    x_train, x_test = standardize(x_train, x_test)
    y_train_s, _, y_mean, y_std = standardize_pose(y_train, y_test)
    model, train_sec, params = train_temporal(x_train, y_train_s, mask_train, names_train, x_test, args)
    pred_s, gates, pred_sec = predict_temporal(model, x_test, names_test, args)
    pred = pred_s * y_std + y_mean
    rows = [
        evaluate("mean_pose", y_test, mean_pose_prediction(y_train, mask_train, len(y_test)), mask_test, 0.0, 0.0, 0),
        evaluate("piw3d_temporal_cp_saff_query", y_test, pred, mask_test, train_sec, pred_sec, params, gates),
    ]
    args.output.parent.mkdir(exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"saved_csv={args.output}")


if __name__ == "__main__":
    main()
