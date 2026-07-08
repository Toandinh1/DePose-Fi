import argparse
import csv
from itertools import permutations
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import nonnegative_cp_mu  # noqa: E402


LINKS = 9
BASE_SUBCARRIERS = 30
PACKETS = 20
JOINTS = 14
MAX_PEOPLE = 3

# 14-keypoint PiW skeleton. This follows a COCO-like upper/lower body tree and
# is used only as a soft anatomical regularizer.
BONES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 13),
)


def require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "h5py is required because Person-in-WiFi 3D CSI files are MATLAB v7.3/HDF5. "
            "Install with `pip install h5py` and rerun."
        ) from exc
    return h5py


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for CP+S-AFF training.") from exc
    return torch, nn, DataLoader, TensorDataset


def read_names(split_root, mode):
    list_path = split_root / f"{mode}_data_list.txt"
    return [line.strip().split()[0] for line in list_path.read_text().splitlines() if line.strip()]


def csi_sanitization(csi_rx):
    one_csi = csi_rx[0, :, :]
    two_csi = csi_rx[1, :, :]
    three_csi = csi_rx[2, :, :]
    pi = np.pi
    antennas = 3
    subcarriers = BASE_SUBCARRIERS
    packets = one_csi.shape[1]
    fi = 312.5 * 2
    csi_phase = np.zeros((antennas, subcarriers, packets), dtype=np.float32)
    ai = np.tile(2 * pi * fi * np.arange(subcarriers), antennas)
    bi = np.ones(antennas * subcarriers)
    a_dot = np.dot(ai, ai)
    b_dot = np.dot(ai, bi)
    c_dot = np.dot(bi, bi)
    temp = np.tile(np.arange(subcarriers), antennas).reshape(antennas, subcarriers)
    for t in range(packets):
        csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
        csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
        csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
        ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
        d_dot = np.dot(ai, ci)
        e_dot = np.dot(bi, ci)
        rho_opt = (b_dot * e_dot - c_dot * d_dot) / (a_dot * c_dot - b_dot ** 2)
        beta_opt = (b_dot * d_dot - a_dot * e_dot) / (a_dot * c_dot - b_dot ** 2)
        csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
    antenna_one = np.abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
    antenna_two = np.abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
    antenna_three = np.abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
    return np.stack([antenna_one, antenna_two, antenna_three], axis=0)


def phase_deno(csi):
    return np.stack([csi_sanitization(csi[rx, :, :, :]) for rx in range(csi.shape[0])], axis=0)


def load_piw_csi(frame_path, feature_mode):
    h5py = require_h5py()
    with h5py.File(frame_path, "r") as f:
        csi = f["csi_out"]
        arr = csi["real"][:] + 1j * csi["imag"][:]
    # Official loader: csi_out -> transpose(3, 2, 1, 0), shape 3 x 3 x 30 x 20.
    arr = arr.transpose(3, 2, 1, 0)
    amp = np.abs(arr).astype(np.float32).reshape(LINKS, BASE_SUBCARRIERS, PACKETS)
    amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(amp.min())
    mx = float(amp.max())
    if mx > mn:
        amp = (amp - mn) / (mx - mn)
    if feature_mode == "amp":
        return amp
    if feature_mode == "amp_sanitized_phase":
        phase_src = phase_deno(arr)
    else:
        phase_src = arr
    phase = np.angle(phase_src).astype(np.float32).reshape(LINKS, BASE_SUBCARRIERS, PACKETS)
    phase = (phase + np.pi) / (2.0 * np.pi)
    return np.concatenate([amp, phase], axis=1).astype(np.float32)


def load_piw_keypoints(path, max_people=MAX_PEOPLE):
    keypoints = np.load(path).astype(np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (JOINTS, 3):
        raise ValueError(f"Expected N x {JOINTS} x 3 keypoints, got {keypoints.shape} in {path}")
    n = min(len(keypoints), max_people)
    padded = np.zeros((max_people, JOINTS, 3), dtype=np.float32)
    mask = np.zeros((max_people,), dtype=np.float32)
    padded[:n] = keypoints[:n]
    mask[:n] = 1.0
    return padded, mask


def cp_feature(frame, rank, iters):
    a, b, c, _ = nonnegative_cp_mu(frame, rank=rank, iters=iters, seed=rank)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()]).astype(np.float32)


def raw_stats_feature(frame):
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


def featurize_split(split_root, mode, rank, iters, feature_mode, max_samples=None, sample_seed=None, progress_every=1000):
    names = read_names(split_root, mode)
    if max_samples is not None:
        if sample_seed is None:
            names = names[:max_samples]
        else:
            rng = np.random.RandomState(sample_seed)
            idx = rng.choice(len(names), size=min(max_samples, len(names)), replace=False)
            names = [names[i] for i in sorted(idx.tolist())]
    x_cp = []
    x_stats = []
    y = []
    mask = []
    used = []
    t0 = time.time()
    for i, name in enumerate(names, start=1):
        csi_path = split_root / "csi" / f"{name}.mat"
        kp_path = split_root / "keypoint" / f"{name}.npy"
        if not csi_path.exists() or not kp_path.exists():
            continue
        frame = load_piw_csi(csi_path, feature_mode)
        keypoints, people_mask = load_piw_keypoints(kp_path)
        x_cp.append(cp_feature(frame, rank, iters))
        x_stats.append(raw_stats_feature(frame))
        y.append(keypoints)
        mask.append(people_mask)
        used.append(name)
        if i % progress_every == 0:
            print(f"{mode} featurized={i}/{len(names)} used={len(used)} elapsed_sec={time.time() - t0:.1f}", flush=True)
    return (
        np.vstack(x_cp).astype(np.float32),
        np.vstack(x_stats).astype(np.float32),
        np.stack(y).astype(np.float32),
        np.stack(mask).astype(np.float32),
        np.array(used),
    )


def standardize(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps)


def standardize_pose(train, test, eps=1e-6):
    valid = np.abs(train).sum(axis=(2, 3)) > 0
    values = train[valid]
    mean = values.mean(axis=(0, 1), keepdims=True).reshape(1, 1, 1, 3)
    std = values.std(axis=(0, 1), keepdims=True).reshape(1, 1, 1, 3)
    return (train - mean) / (std + eps), (test - mean) / (std + eps), mean, std


def make_saff(nn, rank, out_dim, temperature, subcarriers, model_size):
    torch, _, _, _ = require_torch()
    a_size = LINKS * rank
    b_size = subcarriers * rank
    if model_size == "small":
        a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = 32, 96, 32, 96, 32, 16
    elif model_size == "medium":
        a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = 64, 192, 64, 192, 64, 32
    else:
        a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = 96, 256, 96, 256, 96, 48

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
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
            self.heads = nn.ModuleList(
                [nn.Linear(a_dim, out_dim), nn.Linear(b_dim, out_dim), nn.Linear(c_dim, out_dim), nn.Linear(f_dim, out_dim)]
            )

        def forward(self, x, return_gates=False):
            a = x[:, :a_size].reshape((-1, rank, LINKS))
            b = x[:, a_size : a_size + b_size].reshape((-1, rank, subcarriers))
            c = x[:, a_size + b_size :].reshape((-1, rank, PACKETS))
            fa = self.a_net(a)
            fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
            fc = self.c_net(c)
            h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
            ff = self.fuse_net(h)
            gates = torch.softmax(self.gate(h) / self.temperature, dim=1)
            preds = torch.stack(
                [self.heads[0](fa), self.heads[1](fb), self.heads[2](fc), self.heads[3](ff)],
                dim=1,
            )
            out = torch.sum(gates.unsqueeze(-1) * preds, dim=1)
            if return_gates:
                return out, gates
            return out

    return Net()


def saff_dims(model_size):
    if model_size == "small":
        return 32, 96, 32, 96, 32, 16
    if model_size == "medium":
        return 64, 192, 64, 192, 64, 32
    return 96, 256, 96, 256, 96, 48


def make_query_saff(nn, rank, temperature, subcarriers, model_size, num_queries, query_mixer):
    torch, _, _, _ = require_torch()
    a_size = LINKS * rank
    b_size = subcarriers * rank
    a_dim, b_dim, c_dim, f_dim, b_channels, c_channels = saff_dims(model_size)
    hidden = f_dim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
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
            self.query_embed = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
            self.query_net = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.query_mixer = query_mixer
            if query_mixer == "gru":
                self.mixer = nn.GRU(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
            elif query_mixer == "attention":
                heads = 4 if hidden % 4 == 0 else 2
                self.mixer = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.mixer_norm = nn.LayerNorm(hidden)
            self.pose_head = nn.Linear(hidden, JOINTS * 3)
            self.cls_head = nn.Linear(hidden, 1)
            self.count_head = nn.Linear(hidden, MAX_PEOPLE + 1)

        def forward(self, x, return_gates=False):
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
            q = base.unsqueeze(1) + self.query_embed.unsqueeze(0)
            q = self.query_net(q)
            if self.query_mixer == "gru":
                q, _ = self.mixer(q)
            elif self.query_mixer == "attention":
                mixed, _ = self.mixer(q, q, q, need_weights=False)
                q = self.mixer_norm(q + mixed)
            poses = self.pose_head(q).reshape((-1, num_queries, JOINTS, 3))
            logits = self.cls_head(q).squeeze(-1)
            count_logits = self.count_head(base)
            if return_gates:
                return poses, logits, count_logits, gates
            return poses, logits, count_logits

    return Net()


def gate_entropy(gates, eps=1e-8):
    return -(gates * (gates + eps).log()).sum(dim=1).mean()


def permutation_mse_loss(pred, target, mask, torch):
    b, k, j, c = target.shape
    pred = pred.reshape(b, k, j, c)
    perms = list(permutations(range(k)))
    losses = []
    weights = mask[:, :, None, None]
    denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0) * j * c
    for perm in perms:
        p = pred[:, perm]
        mse = ((p - target) ** 2 * weights).sum(dim=(1, 2, 3)) / denom
        losses.append(mse)
    return torch.stack(losses, dim=1).min(dim=1).values.mean()


_ASSIGNMENT_CACHE = {}


def ordered_assignments(num_queries, num_people):
    key = (num_queries, num_people)
    if key not in _ASSIGNMENT_CACHE:
        _ASSIGNMENT_CACHE[key] = list(permutations(range(num_queries), num_people))
    return _ASSIGNMENT_CACHE[key]


def pose_cost_tensor(pred, target, loss_type, torch):
    diff = pred - target
    if loss_type == "l1":
        return diff.abs().mean(dim=(3, 4))
    if loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=(3, 4))
    return (diff ** 2).mean(dim=(3, 4))


def bone_length_loss(pred, target, torch):
    losses = []
    for j1, j2 in BONES:
        pred_len = torch.linalg.norm(pred[:, j1] - pred[:, j2], dim=-1)
        target_len = torch.linalg.norm(target[:, j1] - target[:, j2], dim=-1)
        losses.append((pred_len - target_len).abs())
    return torch.stack(losses, dim=1).mean()


def query_set_loss(poses, logits, count_logits, target, mask, torch, cls_weight, count_weight, bone_weight, pose_loss_type):
    b, q, j, c = poses.shape
    total_pose = poses.new_tensor(0.0)
    total_cls = poses.new_tensor(0.0)
    groups = 0
    counts = mask.sum(dim=1).long()
    for n in range(1, MAX_PEOPLE + 1):
        idx = torch.where(counts == n)[0]
        if idx.numel() == 0:
            continue
        p = poses[idx]
        t = target[idx, :n]
        l = logits[idx]
        # [B, Q, N] query-to-ground-truth pose cost.
        cost = pose_cost_tensor(p[:, :, None], t[:, None], pose_loss_type, torch)
        assigns = torch.tensor(ordered_assignments(q, n), dtype=torch.long, device=poses.device)
        assign_cost = cost[:, assigns, torch.arange(n, device=poses.device)].sum(dim=2)
        best = assign_cost.argmin(dim=1)
        chosen = assigns[best]
        matched_pose_loss = cost[
            torch.arange(idx.numel(), device=poses.device).unsqueeze(1), chosen, torch.arange(n, device=poses.device)
        ]
        matched_pose_loss = matched_pose_loss.mean()
        if bone_weight > 0:
            chosen_pose = p[torch.arange(idx.numel(), device=poses.device).unsqueeze(1), chosen]
            bone_loss = bone_length_loss(chosen_pose.reshape(-1, j, c), t.reshape(-1, j, c), torch)
        else:
            bone_loss = poses.new_tensor(0.0)
        cls_target = torch.zeros_like(l)
        cls_target[torch.arange(idx.numel(), device=poses.device).unsqueeze(1), chosen] = 1.0
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(l, cls_target)
        total_pose = total_pose + matched_pose_loss + bone_weight * bone_loss
        total_cls = total_cls + cls_loss
        groups += 1
    if groups == 0:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    counts = mask.sum(dim=1).long().clamp(0, MAX_PEOPLE)
    count_loss = torch.nn.functional.cross_entropy(count_logits, counts)
    return total_pose / groups + cls_weight * total_cls / groups + count_weight * count_loss


def infer_subcarriers(feature_dim, rank):
    modes = feature_dim // rank
    subcarriers = modes - LINKS - PACKETS
    if feature_dim % rank != 0 or subcarriers <= 0:
        raise ValueError(f"Cannot infer CP dimensions from feature_dim={feature_dim}, rank={rank}")
    return subcarriers


def train_saff(x_train, y_train, mask_train, x_test, args):
    torch, nn, DataLoader, TensorDataset = require_torch()
    device = torch.device(args.device)
    out_dim = MAX_PEOPLE * JOINTS * 3
    subcarriers = infer_subcarriers(x_train.shape[1], args.rank)
    model = make_saff(nn, args.rank, out_dim, args.gate_temperature, subcarriers, args.model_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(mask_train.astype(np.float32)),
    )
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
            pred, gates = model(xb, return_gates=True)
            loss = permutation_mse_loss(pred, yb, mb, torch)
            if args.gate_entropy_weight > 0:
                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0
    model.eval()
    preds = []
    gates_all = []
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size].astype(np.float32)).to(device)
            pred, gates = model(xb, return_gates=True)
            preds.append(pred.cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    pred_sec = time.time() - t1
    params = sum(p.numel() for p in model.parameters())
    return np.vstack(preds).reshape((-1, MAX_PEOPLE, JOINTS, 3)), np.vstack(gates_all), train_sec, pred_sec, params


def train_query_saff(x_train, y_train, mask_train, x_test, args):
    torch, nn, DataLoader, TensorDataset = require_torch()
    device = torch.device(args.device)
    subcarriers = infer_subcarriers(x_train.shape[1], args.rank)
    model = make_query_saff(
        nn, args.rank, args.gate_temperature, subcarriers, args.model_size, args.num_queries, args.query_mixer
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(mask_train.astype(np.float32)),
    )
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
    model.eval()
    preds = []
    scores = []
    gates_all = []
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size].astype(np.float32)).to(device)
            poses, logits, _, gates = model(xb, return_gates=True)
            preds.append(poses.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    pred_sec = time.time() - t1
    params = sum(p.numel() for p in model.parameters())
    pred = np.vstack(preds)
    score = np.vstack(scores)
    top = np.argsort(-score, axis=1)[:, :MAX_PEOPLE]
    pred_top = np.take_along_axis(pred, top[:, :, None, None], axis=1)
    return pred_top, np.vstack(gates_all), train_sec, pred_sec, params


def matched_mpjpe_mm(y_true, y_pred, mask):
    perms = list(permutations(range(MAX_PEOPLE)))
    per_sample = []
    for yt, yp, m in zip(y_true, y_pred, mask):
        n = int(m.sum())
        if n == 0:
            continue
        best = None
        for perm in perms:
            pred_sel = yp[list(perm)[:n]]
            err = np.linalg.norm(yt[:n] - pred_sel, axis=-1).mean()
            if best is None or err < best:
                best = err
        per_sample.append(best * 1000.0)
    return float(np.mean(per_sample))


def pck_mm(y_true, y_pred, mask, threshold_mm):
    perms = list(permutations(range(MAX_PEOPLE)))
    hits = []
    threshold = threshold_mm / 1000.0
    for yt, yp, m in zip(y_true, y_pred, mask):
        n = int(m.sum())
        if n == 0:
            continue
        best_err = None
        for perm in perms:
            pred_sel = yp[list(perm)[:n]]
            err = np.linalg.norm(yt[:n] - pred_sel, axis=-1)
            if best_err is None or err.mean() < best_err.mean():
                best_err = err
        hits.append(best_err <= threshold)
    return float(100.0 * np.concatenate([h.ravel() for h in hits]).mean())


def summarize_gates(gates):
    out = {
        "gate_entropy": float((-gates * np.log(gates + 1e-8)).sum(axis=1).mean()),
        "gate_max_mean": float(gates.max(axis=1).mean()),
    }
    for idx, name in enumerate(["a", "b", "c", "fused"]):
        out[f"gate_mean_{name}"] = float(gates[:, idx].mean())
        out[f"gate_choice_{name}_pct"] = float(100.0 * np.mean(gates.argmax(axis=1) == idx))
    return out


def evaluate(name, y_true, y_pred, mask, train_sec, pred_sec, params, gates=None):
    row = {
        "name": name,
        "mpjpe_mm": matched_mpjpe_mm(y_true, y_pred, mask),
        "pck_50mm": pck_mm(y_true, y_pred, mask, 50),
        "pck_100mm": pck_mm(y_true, y_pred, mask, 100),
        "pck_150mm": pck_mm(y_true, y_pred, mask, 150),
        "train_sec": train_sec,
        "predict_sec": pred_sec,
        "us_per_sample": 1e6 * pred_sec / len(y_true),
        "params": params,
    }
    if gates is not None:
        row.update(summarize_gates(gates))
    counts = mask.sum(axis=1).astype(int)
    for n in range(1, MAX_PEOPLE + 1):
        idx = counts == n
        row[f"n{n}_samples"] = int(idx.sum())
        if idx.any():
            row[f"n{n}_mpjpe_mm"] = matched_mpjpe_mm(y_true[idx], y_pred[idx], mask[idx])
            row[f"n{n}_pck_100mm"] = pck_mm(y_true[idx], y_pred[idx], mask[idx], 100)
    return row


def mean_pose_prediction(y_train, mask_train, n_test):
    valid = mask_train > 0
    mean_person = y_train[valid].mean(axis=0)
    return np.repeat(mean_person[None, None, :, :], n_test * MAX_PEOPLE, axis=0).reshape(
        n_test, MAX_PEOPLE, JOINTS, 3
    )


def frame_id(name):
    parts = str(name).split("_")
    if len(parts) < 3:
        return str(name), None
    try:
        return "_".join(parts[:-1]), int(parts[-1])
    except ValueError:
        return "_".join(parts[:-1]), None


def temporal_smooth_features(x, names, radius):
    if radius <= 0 or names is None:
        return x
    lookup = {}
    parsed = []
    for idx, name in enumerate(names):
        seq, t = frame_id(name)
        parsed.append((seq, t))
        if t is not None:
            lookup[(seq, t)] = idx
    out = np.empty_like(x)
    for idx, (seq, t) in enumerate(parsed):
        if t is None:
            out[idx] = x[idx]
            continue
        xs = [x[idx]]
        weights = [1.0]
        for d in range(1, radius + 1):
            w = 1.0 / (d + 1.0)
            for sign in (-1, 1):
                j = lookup.get((seq, t + sign * d))
                if j is not None:
                    xs.append(x[j])
                    weights.append(w)
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / weights.sum()
        out[idx] = np.sum(np.stack(xs, axis=0) * weights[:, None], axis=0)
    print(f"temporal_smooth radius={radius}", flush=True)
    return out


def filter_by_people(x_train, y_train, mask_train, train_names, x_test, y_test, mask_test, test_names, people_filter):
    if people_filter <= 0:
        return x_train, y_train, mask_train, train_names, x_test, y_test, mask_test, test_names
    train_idx = mask_train.sum(axis=1).astype(int) == people_filter
    test_idx = mask_test.sum(axis=1).astype(int) == people_filter
    if not train_idx.any() or not test_idx.any():
        raise SystemExit(f"No samples found for --people-filter {people_filter}")
    print(
        f"people_filter={people_filter} train={int(train_idx.sum())}/{len(train_idx)} "
        f"test={int(test_idx.sum())}/{len(test_idx)}",
        flush=True,
    )
    return (
        x_train[train_idx],
        y_train[train_idx],
        mask_train[train_idx],
        train_names[train_idx] if train_names is not None else None,
        x_test[test_idx],
        y_test[test_idx],
        mask_test[test_idx],
        test_names[test_idx] if test_names is not None else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "PersonInWiFi3D")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "piw3d_cp_features.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "piw3d_cp_saff.csv")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--feature-mode", choices=["amp", "amp_phase", "amp_sanitized_phase"], default="amp_sanitized_phase")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--head", choices=["fixed", "query"], default="fixed")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--bone-weight", type=float, default=0.0)
    parser.add_argument("--pose-loss", choices=["mse", "l1", "smooth_l1"], default="mse")
    parser.add_argument("--query-mixer", choices=["none", "gru", "attention"], default="none")
    parser.add_argument("--people-filter", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--temporal-radius", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_root = args.data_root / "train_data"
    test_root = args.data_root / "test_data"
    if not train_root.exists() or not test_root.exists():
        raise SystemExit(f"Expected train_data/test_data under {args.data_root}")

    cache_ok = args.cache.exists() and not args.rebuild_cache
    if cache_ok:
        z = np.load(args.cache)
        x_train = z["x_cp_train"]
        x_test = z["x_cp_test"]
        y_train = z["y_train"]
        y_test = z["y_test"]
        mask_train = z["mask_train"]
        mask_test = z["mask_test"]
        train_names = z["train_names"] if "train_names" in z.files else None
        test_names = z["test_names"] if "test_names" in z.files else None
    else:
        x_train, _, y_train, mask_train, train_names = featurize_split(
            train_root, "train", args.rank, args.cp_iters, args.feature_mode, args.max_train, args.sample_seed
        )
        x_test, _, y_test, mask_test, test_names = featurize_split(
            test_root, "test", args.rank, args.cp_iters, args.feature_mode, args.max_test, args.sample_seed
        )
        args.cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            args.cache,
            x_cp_train=x_train,
            x_cp_test=x_test,
            y_train=y_train,
            y_test=y_test,
            mask_train=mask_train,
            mask_test=mask_test,
            train_names=train_names,
            test_names=test_names,
            feature_mode=args.feature_mode,
        )
        print(f"saved_cache={args.cache}")

    x_train, y_train, mask_train, train_names, x_test, y_test, mask_test, test_names = filter_by_people(
        x_train, y_train, mask_train, train_names, x_test, y_test, mask_test, test_names, args.people_filter
    )
    x_train = temporal_smooth_features(x_train, train_names, args.temporal_radius)
    x_test = temporal_smooth_features(x_test, test_names, args.temporal_radius)
    x_train, x_test = standardize(x_train, x_test)
    y_train_s, y_test_s, y_mean, y_std = standardize_pose(y_train, y_test)
    rows = [evaluate("mean_pose", y_test, mean_pose_prediction(y_train, mask_train, len(y_test)), mask_test, 0.0, 0.0, 0)]

    if args.head == "query":
        pred_s, gates, train_sec, pred_sec, params = train_query_saff(x_train, y_train_s, mask_train, x_test, args)
        method_name = "piw3d_cp_saff_query"
    else:
        pred_s, gates, train_sec, pred_sec, params = train_saff(x_train, y_train_s, mask_train, x_test, args)
        method_name = "piw3d_cp_saff"
    pred = pred_s * y_std + y_mean
    rows.append(evaluate(method_name, y_test, pred, mask_test, train_sec, pred_sec, params, gates))

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
