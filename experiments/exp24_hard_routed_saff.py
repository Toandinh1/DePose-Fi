"""Hard-routed S-AFF experiment.

This tests a real deployment route for the "parallel/distributed" architecture
idea: train the normal S-AFF model, then use its sharp gates as an inference-time
routing policy that executes fewer pose heads.

Modes:
  - full_soft: original weighted S-AFF output.
  - top1/top2/top3: keep only the top-k gated experts and renormalize.
  - threshold: use top-1 when gate confidence is high; otherwise full_soft.

This is a deployment tradeoff experiment: accuracy vs executed experts.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "experiments"))

from exp14_cp_cnn_aff import (  # noqa: E402
    LINKS,
    OUT_DIM,
    PACKETS,
    RANK,
    SUBCARRIERS,
    cp_as_component_image,
    evaluate,
    gate_entropy,
    standardize,
    standardize_y,
)


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required.") from exc
    return torch, nn, DataLoader, TensorDataset


def make_routable_saff(nn, temperature):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = temperature
            self.a_net = nn.Sequential(nn.Flatten(), nn.Linear(RANK * LINKS, 32), nn.ReLU())
            self.b_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(RANK, RANK), nn.Sigmoid())
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
            self.fuse_net = nn.Sequential(nn.Linear(32 + 96 + 32, 96), nn.ReLU())
            self.gate = nn.Linear(32 + 96 + 32, 4)
            self.heads = nn.ModuleList(
                [nn.Linear(32, OUT_DIM), nn.Linear(96, OUT_DIM), nn.Linear(32, OUT_DIM), nn.Linear(96, OUT_DIM)]
            )

        def features(self, x):
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
            preds = torch.stack([self.heads[0](fa), self.heads[1](fb), self.heads[2](fc), self.heads[3](ff)], dim=1)
            return preds, gates

        def forward(self, x, return_gates=False):
            preds, gates = self.features(x)
            out = torch.sum(gates.unsqueeze(-1) * preds, dim=1)
            if return_gates:
                return out, gates
            return out

        def route(self, x, mode, threshold=0.85):
            preds, gates = self.features(x)
            if mode == "full_soft":
                return torch.sum(gates.unsqueeze(-1) * preds, dim=1), gates
            if mode.startswith("top"):
                k = int(mode.replace("top", ""))
                vals, idx = torch.topk(gates, k=k, dim=1)
                mask = torch.zeros_like(gates).scatter_(1, idx, vals)
                mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
                return torch.sum(mask.unsqueeze(-1) * preds, dim=1), gates
            if mode == "threshold":
                vals, idx = torch.max(gates, dim=1, keepdim=True)
                top1 = torch.zeros_like(gates).scatter_(1, idx, 1.0)
                weights = torch.where(vals >= threshold, top1, gates)
                return torch.sum(weights.unsqueeze(-1) * preds, dim=1), gates
            raise ValueError(f"unknown route mode: {mode}")

    return Net()


def gate_stats(gates):
    eps = 1e-8
    entropy = -np.sum(gates * np.log(gates + eps), axis=1)
    out = {
        "gate_entropy": float(entropy.mean()),
        "gate_max_mean": float(gates.max(axis=1).mean()),
    }
    names = ["a", "b", "c", "fused"]
    for i, name in enumerate(names):
        out[f"gate_choice_{name}_pct"] = float(100 * np.mean(gates.argmax(axis=1) == i))
        out[f"gate_mean_{name}"] = float(gates[:, i].mean())
    return out


def executed_experts(gates, mode, threshold):
    if mode == "full_soft":
        return 4.0, 100.0
    if mode.startswith("top"):
        return float(int(mode.replace("top", ""))), 100.0
    if mode == "threshold":
        confident = gates.max(axis=1) >= threshold
        return float(np.mean(np.where(confident, 1, 4))), float(100 * confident.mean())
    raise ValueError(mode)


def train(model, torch, DataLoader, TensorDataset, x_train, y_train, args):
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            pred, gates = model(xb, return_gates=True)
            loss = loss_fn(pred, yb)
            if args.gate_entropy_weight > 0:
                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    return time.time() - t0


def predict_mode(model, torch, x_test, mode, batch_size, device, threshold):
    preds, gates_all = [], []
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            xb = torch.from_numpy(x_test[start : start + batch_size]).to(device)
            pred, gates = model.route(xb, mode, threshold)
            preds.append(pred.cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    return np.vstack(preds), np.vstack(gates_all), time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "protocol3_frame_cp_probe_full_cp10.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "hard_routed_saff.csv")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--thresholds", default="0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    z = np.load(args.input)
    x_train = z["x_cp_train"]
    x_test = z["x_cp_test"]
    y_train = z["y_train"].astype(np.float32)
    y_test = z["y_test"].astype(np.float32)
    if args.max_train is not None:
        x_train, y_train = x_train[: args.max_train], y_train[: args.max_train]
    if args.max_test is not None:
        x_test, y_test = x_test[: args.max_test], y_test[: args.max_test]

    x_train = cp_as_component_image(x_train)
    x_test = cp_as_component_image(x_test)
    x_train, x_test = standardize(x_train, x_test)
    y_train_s, _, y_mean, y_std = standardize_y(y_train, y_test)

    model = make_routable_saff(nn, args.gate_temperature)
    params = sum(p.numel() for p in model.parameters())
    train_sec = train(model, torch, DataLoader, TensorDataset, x_train, y_train_s, args)

    rows = []
    route_jobs = [("full_soft", None), ("top1", None), ("top2", None), ("top3", None)]
    for t in [float(v) for v in args.thresholds.split(",")]:
        route_jobs.append((f"threshold_{t:.2f}", t))

    for mode_name, threshold in route_jobs:
        mode = "threshold" if threshold is not None else mode_name
        pred_s, gates, pred_sec = predict_mode(
            model, torch, x_test, mode, args.batch_size, args.device, threshold or 0.0
        )
        pred = pred_s * y_std + y_mean
        row = evaluate(mode_name, y_test, pred, train_sec, pred_sec, params, gate_stats(gates))
        avg_experts, confident_pct = executed_experts(gates, mode, threshold or 0.0)
        row.update(
            {
                "avg_executed_experts": avg_experts,
                "expert_compute_fraction": avg_experts / 4.0,
                "estimated_expert_speedup": 4.0 / avg_experts,
                "threshold_confident_pct": confident_pct,
            }
        )
        rows.append(row)
        print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={args.output}")


if __name__ == "__main__":
    main()
