import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

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
OUT_DIM = 17 * 3


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install it first, e.g. `pip install torch`, "
            "then rerun this experiment."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def split_abc(x):
    a = x[:, :A_SIZE].reshape((-1, LINKS, RANK)).transpose(0, 2, 1)
    b = x[:, A_SIZE : A_SIZE + B_SIZE].reshape((-1, SUBCARRIERS, RANK)).transpose(0, 2, 1)
    c = x[:, A_SIZE + B_SIZE :].reshape((-1, PACKETS, RANK)).transpose(0, 2, 1)
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def cp_as_component_image(x):
    a, b, c = split_abc(x)
    z = np.concatenate([a, b, c], axis=2)
    return z[:, None, :, :].astype(np.float32)  # N x 1 x R x 127


def standardize(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps)


def standardize_y(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (train - mean) / (std + eps), (test - mean) / (std + eps), mean, std


class CPCNN:
    def __init__(self, nn):
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)),
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, OUT_DIM),
        )

    def __call__(self):
        return self.net


class CPAFF:
    """WiDeFus-inspired adaptive fusion over CP factor modes.

    This adapts the idea, not the exact HAR module:
    - link branch processes A,
    - subcarrier branch processes B with lightweight 1D CNN attention,
    - packet branch processes C,
    - learnable gate fuses branch embeddings for pose regression.
    """

    def __init__(self, nn):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a_net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(RANK * LINKS, 32),
                    nn.ReLU(),
                )
                self.b_att = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(RANK, RANK),
                    nn.Sigmoid(),
                )
                self.b_net = nn.Sequential(
                    nn.Conv1d(RANK, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),
                    nn.Flatten(),
                    nn.Linear(32 * 8, 96),
                    nn.ReLU(),
                )
                self.c_net = nn.Sequential(
                    nn.Conv1d(RANK, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(4),
                    nn.Flatten(),
                    nn.Linear(16 * 4, 32),
                    nn.ReLU(),
                )
                self.gate = nn.Sequential(
                    nn.Linear(32 + 96 + 32, 3),
                    nn.Softmax(dim=1),
                )
                self.heads = nn.ModuleList(
                    [
                        nn.Linear(32, OUT_DIM),
                        nn.Linear(96, OUT_DIM),
                        nn.Linear(32, OUT_DIM),
                    ]
                )

            def forward(self, x):
                # x: N x 1 x R x 127
                z = x[:, 0]
                a = z[:, :, :LINKS]
                b = z[:, :, LINKS : LINKS + SUBCARRIERS]
                c = z[:, :, LINKS + SUBCARRIERS :]
                fa = self.a_net(a)
                att = self.b_att(b).unsqueeze(-1)
                fb = self.b_net(b * att)
                fc = self.c_net(c)
                gates = self.gate(nn.functional.normalize(torch.cat([fa, fb, fc], dim=1), dim=1))
                ya = self.heads[0](fa)
                yb = self.heads[1](fb)
                yc = self.heads[2](fc)
                return gates[:, 0:1] * ya + gates[:, 1:2] * yb + gates[:, 2:3] * yc

        torch, _, _, _ = require_torch()
        self.net = Net()

    def __call__(self):
        return self.net


class CPSelectiveAFF:
    """Selective AFF with sparse per-sample routing.

    The model keeps the original branch embeddings and lets the gate choose
    among link-only, subcarrier-only, packet-only, and all-branch fused heads.
    An optional entropy loss can sharpen gates toward decisive selections.
    """

    def __init__(self, nn, temperature=1.0):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = temperature
                self.a_net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(RANK * LINKS, 32),
                    nn.ReLU(),
                )
                self.b_att = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(RANK, RANK),
                    nn.Sigmoid(),
                )
                self.b_net = nn.Sequential(
                    nn.Conv1d(RANK, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),
                    nn.Flatten(),
                    nn.Linear(32 * 8, 96),
                    nn.ReLU(),
                )
                self.c_net = nn.Sequential(
                    nn.Conv1d(RANK, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(4),
                    nn.Flatten(),
                    nn.Linear(16 * 4, 32),
                    nn.ReLU(),
                )
                self.fuse_net = nn.Sequential(
                    nn.Linear(32 + 96 + 32, 96),
                    nn.ReLU(),
                )
                self.gate = nn.Linear(32 + 96 + 32, 4)
                self.heads = nn.ModuleList(
                    [
                        nn.Linear(32, OUT_DIM),
                        nn.Linear(96, OUT_DIM),
                        nn.Linear(32, OUT_DIM),
                        nn.Linear(96, OUT_DIM),
                    ]
                )

            def forward(self, x, return_gates=False):
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
                    [
                        self.heads[0](fa),
                        self.heads[1](fb),
                        self.heads[2](fc),
                        self.heads[3](ff),
                    ],
                    dim=1,
                )
                out = torch.sum(gates.unsqueeze(-1) * preds, dim=1)
                if return_gates:
                    return out, gates
                return out

        torch, _, _, _ = require_torch()
        self.net = Net()

    def __call__(self):
        return self.net


class CPTransformer:
    """Small transformer baseline over CP factor tokens.

    Each CP rank contributes three tokens: link, subcarrier, and packet.
    This tests whether attention-based token mixing improves accuracy enough
    to justify its higher deployment cost.
    """

    def __init__(self, nn, d_model=64, nhead=4, layers=2, dim_feedforward=128, dropout=0.1):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a_proj = nn.Linear(LINKS, d_model)
                self.b_proj = nn.Linear(SUBCARRIERS, d_model)
                self.c_proj = nn.Linear(PACKETS, d_model)
                self.mode_embed = nn.Parameter(torch.zeros(3, d_model))
                self.rank_embed = nn.Parameter(torch.zeros(RANK, d_model))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 128),
                    nn.ReLU(),
                    nn.Linear(128, OUT_DIM),
                )

            def forward(self, x):
                # x: N x 1 x R x 127
                z = x[:, 0]
                a = z[:, :, :LINKS]
                b = z[:, :, LINKS : LINKS + SUBCARRIERS]
                c = z[:, :, LINKS + SUBCARRIERS :]
                ta = self.a_proj(a) + self.mode_embed[0] + self.rank_embed
                tb = self.b_proj(b) + self.mode_embed[1] + self.rank_embed
                tc = self.c_proj(c) + self.mode_embed[2] + self.rank_embed
                tokens = torch.stack([ta, tb, tc], dim=2).reshape(x.shape[0], RANK * 3, -1)
                h = self.encoder(tokens).mean(dim=1)
                return self.head(h)

        torch, _, _, _ = require_torch()
        self.net = Net()

    def __call__(self):
        return self.net


def evaluate(name, y_true, y_pred, train_sec, pred_sec, params, gate_stats=None):
    gate_stats = gate_stats or {}
    return {
        "name": name,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
        "train_sec": train_sec,
        "predict_sec": pred_sec,
        "us_per_sample": 1e6 * pred_sec / len(y_true),
        "params": params,
        **gate_stats,
    }


def gate_entropy(gates, eps=1e-8):
    return -(gates * (gates + eps).log()).sum(dim=1).mean()


def gate_balance_loss(gates):
    target = torch.full((gates.shape[1],), 1.0 / gates.shape[1], device=gates.device)
    return torch.mean((gates.mean(dim=0) - target) ** 2)


def forward_with_optional_gates(model, xb):
    try:
        return model(xb, return_gates=True)
    except TypeError:
        return model(xb), None


def summarize_gates(gates_np):
    if gates_np is None:
        return {}
    eps = 1e-8
    entropy = -np.sum(gates_np * np.log(gates_np + eps), axis=1)
    stats = {
        "gate_entropy": float(entropy.mean()),
        "gate_max_mean": float(gates_np.max(axis=1).mean()),
        "gate_choice_a_pct": float(100.0 * np.mean(np.argmax(gates_np, axis=1) == 0)),
        "gate_choice_b_pct": float(100.0 * np.mean(np.argmax(gates_np, axis=1) == 1)),
        "gate_choice_c_pct": float(100.0 * np.mean(np.argmax(gates_np, axis=1) == 2)),
    }
    if gates_np.shape[1] > 3:
        stats["gate_choice_fused_pct"] = float(100.0 * np.mean(np.argmax(gates_np, axis=1) == 3))
    for idx in range(gates_np.shape[1]):
        stats[f"gate_mean_{idx}"] = float(gates_np[:, idx].mean())
    return stats


def train_model(model, torch, nn, DataLoader, TensorDataset, x_train, y_train, x_test, args):
    device = torch.device(args.device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred, gates = forward_with_optional_gates(model, xb)
            loss = loss_fn(pred, yb)
            if gates is not None and args.gate_entropy_weight > 0:
                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            if gates is not None and args.gate_balance_weight > 0:
                loss = loss + args.gate_balance_weight * gate_balance_loss(gates)
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
            xb = torch.from_numpy(x_test[start : start + args.batch_size]).to(device)
            pred, gates = forward_with_optional_gates(model, xb)
            preds.append(pred.cpu().numpy())
            if gates is not None:
                gates_all.append(gates.cpu().numpy())
    pred_sec = time.time() - t1
    gate_stats = summarize_gates(np.vstack(gates_all)) if gates_all else {}
    return np.vstack(preds), train_sec, pred_sec, sum(p.numel() for p in model.parameters()), gate_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "protocol3_frame_cp_probe_full_cp10.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cp_cnn_aff_full_cp10.csv")
    parser.add_argument(
        "--model",
        choices=["cnn", "aff", "aff_selective", "aff_compare", "transformer", "both", "all"],
        default="both",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--gate-balance-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    z = np.load(args.input)
    x_train = z["x_cp_train"]
    x_test = z["x_cp_test"]
    y_train = z["y_train"].astype(np.float32)
    y_test = z["y_test"].astype(np.float32)

    if args.max_train is not None:
        x_train = x_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test is not None:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    x_train = cp_as_component_image(x_train)
    x_test = cp_as_component_image(x_test)
    x_train, x_test = standardize(x_train, x_test)
    y_train_s, _, y_mean, y_std = standardize_y(y_train, y_test)

    jobs = []
    if args.model in ["cnn", "both", "all"]:
        jobs.append(("cp_components_cnn", CPCNN(nn)()))
    if args.model in ["aff", "aff_compare", "both", "all"]:
        jobs.append(("cp_components_aff", CPAFF(nn)()))
    if args.model in ["aff_selective", "aff_compare", "all"]:
        jobs.append(("cp_components_selective_aff", CPSelectiveAFF(nn, args.gate_temperature)()))
    if args.model in ["transformer", "all"]:
        jobs.append(("cp_components_transformer", CPTransformer(nn)()))

    rows = []
    for name, model in jobs:
        print(f"training {name}", flush=True)
        pred_s, train_sec, pred_sec, params, gate_stats = train_model(
            model, torch, nn, DataLoader, TensorDataset, x_train, y_train_s, x_test, args
        )
        pred = pred_s * y_std + y_mean
        res = evaluate(name, y_test, pred, train_sec, pred_sec, params, gate_stats)
        rows.append(res)
        print(res, flush=True)

    args.output.parent.mkdir(exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={args.output}")


if __name__ == "__main__":
    main()
