"""Option B: a single truncatable "anytime" CP+S-AFF for Proactive Rank Adaptation.

Motivation
----------
Proactive Rank Adaptation (PRA) treats CP rank as a runtime knob. For the
"nested / zero-switching-cost anytime" claim to be true, the ranks must share
factors: rank-R inference must reuse the rank-(R-1) components plus one more.

Standard non-negative CP-ALS fit independently per rank is NOT nested. This
experiment instead fits CP once at RANK_MAX, orders each frame's components by
energy (so a prefix = the top-R components), and trains ONE S-AFF that is robust
to being evaluated at any prefix rank via component-count dropout. The same model
is then evaluated at R in {2,4,6,8}, giving a self-consistent accuracy model
A(R) for the PRA controller and the contention experiment (exp30).

Only the CP extraction cost scales with R; the S-AFF sees zero-padded tail
components, so parameter count is constant. This matches the paper claim that
CP extraction, not the fusion head, is the rank-sensitive cost.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402

LINKS = 3
SUBCARRIERS = 114
PACKETS = 10
FEAT = LINKS + SUBCARRIERS + PACKETS  # 127
OUT_DIM = 17 * 3
RANK_MAX = 8
CANDIDATE_RANKS = [2, 4, 6, 8]


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed. `pip install torch` then rerun.") from exc
    return torch, nn, DataLoader, TensorDataset


def split_abc(x, rank):
    a_size = LINKS * rank
    b_size = SUBCARRIERS * rank
    a = x[:, :a_size].reshape((-1, LINKS, rank)).transpose(0, 2, 1)
    b = x[:, a_size : a_size + b_size].reshape((-1, SUBCARRIERS, rank)).transpose(0, 2, 1)
    c = x[:, a_size + b_size :].reshape((-1, PACKETS, rank)).transpose(0, 2, 1)
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def cp_component_image(x, rank):
    """Return N x 1 x RANK_MAX x FEAT image with components sorted by energy desc."""
    a, b, c = split_abc(x, rank)  # each N x rank x mode
    z = np.concatenate([a, b, c], axis=2)  # N x rank x FEAT
    # energy of rank-1 term r ~ ||a_r|| * ||b_r|| * ||c_r||
    na = np.linalg.norm(a, axis=2)
    nb = np.linalg.norm(b, axis=2)
    nc = np.linalg.norm(c, axis=2)
    energy = na * nb * nc  # N x rank
    order = np.argsort(-energy, axis=1)  # descending
    z = np.take_along_axis(z, order[:, :, None], axis=1)
    if rank < RANK_MAX:
        pad = np.zeros((z.shape[0], RANK_MAX - rank, FEAT), dtype=np.float32)
        z = np.concatenate([z, pad], axis=1)
    return z[:, None, :, :].astype(np.float32)


def standardize(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps)


def standardize_y(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps), mean, std


def build_model(nn, temperature=1.0):
    class AnytimeSAFF(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
            self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(RANK_MAX * LINKS, 32), nn.ReLU())
            self.b_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(RANK_MAX, RANK_MAX), nn.Sigmoid()
            )
            self.b_net = nn.Sequential(
                nn.Conv1d(RANK_MAX, 32, kernel_size=7, padding=3), nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=7, padding=3), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(32 * 8, 96), nn.ReLU(),
            )
            self.c_net = nn.Sequential(
                nn.Conv1d(RANK_MAX, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(16 * 4, 32), nn.ReLU(),
            )
            self.fuse_net = nn.Sequential(nn.Linear(32 + 96 + 32, 96), nn.ReLU())
            self.gate = nn.Linear(32 + 96 + 32, 4)
            self.heads = nn.ModuleList(
                [nn.Linear(32, OUT_DIM), nn.Linear(96, OUT_DIM),
                 nn.Linear(32, OUT_DIM), nn.Linear(96, OUT_DIM)]
            )

        def forward(self, x, return_gates=False):
            import torch
            z = x[:, 0]
            a = z[:, :, :LINKS]
            b = z[:, :, LINKS : LINKS + SUBCARRIERS]
            c = z[:, :, LINKS + SUBCARRIERS :]
            fa = self.a_net(a)
            fb = self.b_net(b * self.b_att(b).unsqueeze(-1))
            fc = self.c_net(c)
            h = nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1)
            ff = self.fuse_net(h)
            gates = torch.softmax(self.gate(h) / self.temperature, dim=1)
            preds = torch.stack(
                [self.heads[0](fa), self.heads[1](fb), self.heads[2](fc), self.heads[3](ff)], dim=1
            )
            out = torch.sum(gates.unsqueeze(-1) * preds, dim=1)
            if return_gates:
                return out, gates
            return out

    return AnytimeSAFF()


def mask_rank(xb, rank, torch):
    """Zero the tail components beyond `rank` (components are energy-sorted)."""
    if rank >= RANK_MAX:
        return xb
    m = xb.clone()
    m[:, 0, rank:, :] = 0.0
    return m


def per_frame_pck(y_true, y_pred, threshold, eps=1e-8):
    yt = np.asarray(y_true).reshape((-1, 17, 3))[:, :, :2]
    yp = np.asarray(y_pred).reshape((-1, 17, 3))[:, :, :2]
    scale = np.maximum(np.linalg.norm(yt[:, 1, :] - yt[:, 11, :], axis=1), eps)
    dist = np.linalg.norm(yp - yt, axis=2) / scale[:, None]
    return 100.0 * (dist <= threshold).mean(axis=1)  # N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "protocol3_frame_cp_probe_full_cp8.npz")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "anytime_cp_saff_ranks.csv")
    parser.add_argument("--dump", type=Path, default=ROOT / "outputs" / "anytime_pra_rank_eval.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(args.input)
    x_train_raw, x_test_raw = z["x_cp_train"], z["x_cp_test"]
    y_train = z["y_train"].astype(np.float32)
    y_test = z["y_test"].astype(np.float32)

    rank_in = x_train_raw.shape[1] // FEAT
    if rank_in < RANK_MAX:
        raise SystemExit(f"Input CP rank {rank_in} < RANK_MAX {RANK_MAX}; regenerate features at rank {RANK_MAX}.")

    if args.max_train:
        x_train_raw, y_train = x_train_raw[: args.max_train], y_train[: args.max_train]
    if args.max_test:
        x_test_raw, y_test = x_test_raw[: args.max_test], y_test[: args.max_test]

    x_train = cp_component_image(x_train_raw, rank_in)
    x_test = cp_component_image(x_test_raw, rank_in)
    x_train, x_test = standardize(x_train, x_test)
    y_train_s, _, y_mean, y_std = standardize_y(y_train, y_test)

    device = torch.device(args.device)
    model = build_model(nn, args.gate_temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_s))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    params = sum(p.numel() for p in model.parameters())
    print(f"model_params={params} candidate_ranks={CANDIDATE_RANKS}", flush=True)

    t0 = time.time()
    model.train()
    rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # component-count dropout: random target rank per batch
            r = int(rng.choice(CANDIDATE_RANKS))
            xb_r = mask_rank(xb, r, torch)
            opt.zero_grad(set_to_none=True)
            pred = model(xb_r)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0

    # Evaluate the single model at each fixed rank prefix.
    model.eval()
    rows = []
    pck20_by_rank = np.zeros((len(y_test), len(CANDIDATE_RANKS)), dtype=np.float32)
    mpjpe_by_rank = np.zeros((len(y_test), len(CANDIDATE_RANKS)), dtype=np.float32)
    x_test_t = torch.from_numpy(x_test)
    for j, r in enumerate(CANDIDATE_RANKS):
        preds = []
        t1 = time.time()
        with torch.no_grad():
            for start in range(0, len(x_test), args.batch_size):
                xb = x_test_t[start : start + args.batch_size].to(device)
                xb = mask_rank(xb, r, torch)
                preds.append(model(xb).cpu().numpy())
        pred_sec = time.time() - t1
        pred = np.vstack(preds) * y_std + y_mean
        pck20_by_rank[:, j] = per_frame_pck(y_test, pred, 0.2)
        mpjpe_by_rank[:, j] = np.linalg.norm(
            y_test.reshape(-1, 17, 3) - pred.reshape(-1, 17, 3), axis=2
        ).mean(axis=1)
        row = {
            "rank": r,
            "pck_20": hpe_li_pck_mmfi(y_test, pred, 0.2),
            "pck_10": hpe_li_pck_mmfi(y_test, pred, 0.1),
            "pck_30": hpe_li_pck_mmfi(y_test, pred, 0.3),
            "pck_40": hpe_li_pck_mmfi(y_test, pred, 0.4),
            "pck_50": hpe_li_pck_mmfi(y_test, pred, 0.5),
            "mpjpe": mpjpe_3d(y_test, pred),
            "params": params,
            "predict_us_per_sample": 1e6 * pred_sec / len(y_test),
        }
        rows.append(row)
        print(row, flush=True)

    args.csv.parent.mkdir(exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    args.dump.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        args.dump,
        ranks=np.array(CANDIDATE_RANKS),
        pck20_by_rank=pck20_by_rank,
        mpjpe_by_rank=mpjpe_by_rank,
        A_R_pck20=np.array([r["pck_20"] for r in rows], dtype=np.float32),
        params=params,
        train_sec=train_sec,
    )

    summary = {
        "train_sec": train_sec,
        "params": params,
        "A_R_pck20": {r["rank"]: r["pck_20"] for r in rows},
        "A_R_mpjpe": {r["rank"]: r["mpjpe"] for r in rows},
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    print(f"saved_csv={args.csv}")
    print(f"saved_dump={args.dump}")


if __name__ == "__main__":
    main()
