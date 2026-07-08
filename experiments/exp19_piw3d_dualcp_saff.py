import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "src"))

from cp_factorization import nonnegative_cp_mu  # noqa: E402
from exp17_piw3d_cp_saff import (  # noqa: E402
    BASE_SUBCARRIERS,
    JOINTS,
    LINKS,
    MAX_PEOPLE,
    PACKETS,
    evaluate,
    gate_entropy,
    load_piw_csi,
    load_piw_keypoints,
    mean_pose_prediction,
    query_set_loss,
    read_names,
    standardize,
    standardize_pose,
)


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for dual-CP S-AFF training.") from exc
    return torch, nn, DataLoader, TensorDataset


def cp_feature(frame, rank, iters, seed):
    a, b, c, _ = nonnegative_cp_mu(frame, rank=rank, iters=iters, seed=seed)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()]).astype(np.float32)


def load_amp_phase_features(csi_path, rank, iters):
    frame = load_piw_csi(csi_path, "amp_sanitized_phase")
    amp = frame[:, :BASE_SUBCARRIERS, :]
    phase = frame[:, BASE_SUBCARRIERS:, :]
    return cp_feature(amp, rank, iters, seed=rank), cp_feature(phase, rank, iters, seed=rank + 1000)


def featurize_split(split_root, mode, rank, iters, people_filter, max_samples=None, progress_every=1000):
    names = read_names(split_root, mode)
    x_amp, x_phase, y, mask, used = [], [], [], [], []
    t0 = time.time()
    for i, name in enumerate(names, start=1):
        csi_path = split_root / "csi" / f"{name}.mat"
        kp_path = split_root / "keypoint" / f"{name}.npy"
        if not csi_path.exists() or not kp_path.exists():
            continue
        keypoints, people_mask = load_piw_keypoints(kp_path)
        if people_filter > 0 and int(people_mask.sum()) != people_filter:
            continue
        xa, xp = load_amp_phase_features(csi_path, rank, iters)
        x_amp.append(xa)
        x_phase.append(xp)
        y.append(keypoints)
        mask.append(people_mask)
        used.append(name)
        if max_samples is not None and len(used) >= max_samples:
            break
        if i % progress_every == 0:
            print(f"{mode} scanned={i}/{len(names)} used={len(used)} elapsed_sec={time.time() - t0:.1f}", flush=True)
    return (
        np.vstack(x_amp).astype(np.float32),
        np.vstack(x_phase).astype(np.float32),
        np.stack(y).astype(np.float32),
        np.stack(mask).astype(np.float32),
        np.asarray(used),
    )


def saff_dims(model_size):
    if model_size == "small":
        return 32, 96, 32, 96, 32, 16
    if model_size == "medium":
        return 64, 192, 64, 192, 64, 32
    return 96, 256, 96, 256, 96, 48


def make_factor_encoder(nn, rank, model_size):
    torch, _, _, _ = require_torch()
    a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = saff_dims(model_size)
    a_size = LINKS * rank
    b_size = BASE_SUBCARRIERS * rank
    hidden = f_dim

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(rank * LINKS, a_dim), nn.ReLU())
            self.b_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(rank, rank), nn.Sigmoid())
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
            self.proj = nn.ModuleList(
                [nn.Linear(a_dim, hidden), nn.Linear(b_dim, hidden), nn.Linear(c_dim, hidden), nn.Linear(f_dim, hidden)]
            )

        def forward(self, x, temperature):
            a = x[:, :a_size].reshape((-1, rank, LINKS))
            b = x[:, a_size : a_size + b_size].reshape((-1, rank, BASE_SUBCARRIERS))
            c = x[:, a_size + b_size :].reshape((-1, rank, PACKETS))
            fa = self.a_net(a)
            fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
            fc = self.c_net(c)
            h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
            ff = self.fuse_net(h)
            gates = torch.softmax(self.gate(h) / temperature, dim=1)
            branches = torch.stack([self.proj[0](fa), self.proj[1](fb), self.proj[2](fc), self.proj[3](ff)], dim=1)
            base = torch.sum(gates.unsqueeze(-1) * branches, dim=1)
            return base, gates

    return Encoder()


def make_dual_model(nn, rank, model_size, num_queries, query_mixer, temperature):
    torch, _, _, _ = require_torch()
    _, _, _, hidden, _, _ = saff_dims(model_size)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
            self.query_mixer = query_mixer
            self.amp_encoder = make_factor_encoder(nn, rank, model_size)
            self.phase_encoder = make_factor_encoder(nn, rank, model_size)
            self.cross_gate = nn.Sequential(nn.Linear(hidden * 2, 2), nn.Softmax(dim=1))
            self.cross_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.query_embed = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
            self.query_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            if query_mixer == "attention":
                heads = 4 if hidden % 4 == 0 else 2
                self.q_mixer = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.q_norm = nn.LayerNorm(hidden)
            self.pose_head = nn.Linear(hidden, JOINTS * 3)
            self.cls_head = nn.Linear(hidden, 1)
            self.count_head = nn.Linear(hidden, MAX_PEOPLE + 1)

        def forward(self, xa, xp, return_gates=False):
            ha, ga = self.amp_encoder(xa, self.temperature)
            hp, gp = self.phase_encoder(xp, self.temperature)
            both = torch.cat([ha, hp], dim=1)
            stream_gate = self.cross_gate(both)
            fused = self.cross_fuse(both) + stream_gate[:, 0:1] * ha + stream_gate[:, 1:2] * hp
            q = fused.unsqueeze(1) + self.query_embed.unsqueeze(0)
            q = self.query_net(q)
            if self.query_mixer == "attention":
                mixed, _ = self.q_mixer(q, q, q, need_weights=False)
                q = self.q_norm(q + mixed)
            poses = self.pose_head(q).reshape((-1, num_queries, JOINTS, 3))
            logits = self.cls_head(q).squeeze(-1)
            count_logits = self.count_head(fused)
            if return_gates:
                gates = torch.cat([ga, gp, stream_gate], dim=1)
                return poses, logits, count_logits, gates
            return poses, logits, count_logits

    return Net()


def summarize_dual_gates(gates):
    out = {}
    names = ["amp_a", "amp_b", "amp_c", "amp_fused", "phase_a", "phase_b", "phase_c", "phase_fused", "stream_amp", "stream_phase"]
    for i, name in enumerate(names):
        out[f"gate_mean_{name}"] = float(gates[:, i].mean())
    out["amp_gate_entropy"] = float((-gates[:, :4] * np.log(gates[:, :4] + 1e-8)).sum(axis=1).mean())
    out["phase_gate_entropy"] = float((-gates[:, 4:8] * np.log(gates[:, 4:8] + 1e-8)).sum(axis=1).mean())
    out["stream_amp_choice_pct"] = float(100.0 * np.mean(gates[:, 8] >= gates[:, 9]))
    return out


def train_predict(xa_train, xp_train, y_train, mask_train, xa_test, xp_test, args):
    torch, nn, DataLoader, TensorDataset = require_torch()
    device = torch.device(args.device)
    model = make_dual_model(nn, args.rank, args.model_size, args.num_queries, args.query_mixer, args.gate_temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.from_numpy(xa_train.astype(np.float32)),
        torch.from_numpy(xp_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(mask_train.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        for xa, xp, yb, mb in loader:
            xa, xp, yb, mb = xa.to(device), xp.to(device), yb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            poses, logits, count_logits, gates = model(xa, xp, return_gates=True)
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
                loss = loss + args.gate_entropy_weight * (gate_entropy(gates[:, :4]) + gate_entropy(gates[:, 4:8]))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    preds, scores, gates_all = [], [], []
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xa_test), args.batch_size):
            xa = torch.from_numpy(xa_test[start : start + args.batch_size].astype(np.float32)).to(device)
            xp = torch.from_numpy(xp_test[start : start + args.batch_size].astype(np.float32)).to(device)
            poses, logits, _, gates = model(xa, xp, return_gates=True)
            preds.append(poses.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    pred_sec = time.time() - t1
    pred = np.vstack(preds)
    score = np.vstack(scores)
    keep = min(MAX_PEOPLE, pred.shape[1])
    top = np.argsort(-score, axis=1)[:, :keep]
    pred_top = np.take_along_axis(pred, top[:, :, None, None], axis=1)
    if keep < MAX_PEOPLE:
        padded = np.zeros((len(pred_top), MAX_PEOPLE, JOINTS, 3), dtype=pred_top.dtype)
        padded[:, :keep] = pred_top
        pred_top = padded
    return pred_top, np.vstack(gates_all), train_sec, pred_sec, sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "PersonInWiFi3D")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "piw3d_dualcp_1p_rank16_features.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "piw3d_dualcp_saff.csv")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--people-filter", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.cache.exists() and not args.rebuild_cache:
        z = np.load(args.cache)
        xa_train, xp_train = z["x_amp_train"], z["x_phase_train"]
        xa_test, xp_test = z["x_amp_test"], z["x_phase_test"]
        y_train, y_test = z["y_train"], z["y_test"]
        mask_train, mask_test = z["mask_train"], z["mask_test"]
    else:
        train_root = args.data_root / "train_data"
        test_root = args.data_root / "test_data"
        xa_train, xp_train, y_train, mask_train, train_names = featurize_split(
            train_root, "train", args.rank, args.cp_iters, args.people_filter, args.max_train
        )
        xa_test, xp_test, y_test, mask_test, test_names = featurize_split(
            test_root, "test", args.rank, args.cp_iters, args.people_filter, args.max_test
        )
        args.cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            args.cache,
            x_amp_train=xa_train,
            x_phase_train=xp_train,
            x_amp_test=xa_test,
            x_phase_test=xp_test,
            y_train=y_train,
            y_test=y_test,
            mask_train=mask_train,
            mask_test=mask_test,
            train_names=train_names,
            test_names=test_names,
        )
        print(f"saved_cache={args.cache}", flush=True)

    xa_train, xa_test = standardize(xa_train, xa_test)
    xp_train, xp_test = standardize(xp_train, xp_test)
    y_train_s, _, y_mean, y_std = standardize_pose(y_train, y_test)
    pred_s, gates, train_sec, pred_sec, params = train_predict(
        xa_train, xp_train, y_train_s, mask_train, xa_test, xp_test, args
    )
    pred = pred_s * y_std + y_mean
    rows = [
        evaluate("mean_pose", y_test, mean_pose_prediction(y_train, mask_train, len(y_test)), mask_test, 0.0, 0.0, 0),
        evaluate("piw3d_dualcp_saff_query", y_test, pred, mask_test, train_sec, pred_sec, params),
    ]
    rows[-1].update(summarize_dual_gates(gates))
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
