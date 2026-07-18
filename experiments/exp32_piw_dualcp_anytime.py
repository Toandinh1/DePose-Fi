"""Single-person dual-CP anytime S-AFF for Proactive Rank Adaptation (PRA), MPJPE metric.

Builds on exp19 (single-person dual-CP + S-AFF, the S/M/L models that reached
83.82/91.35/107.53 mm) and the anytime recipe of exp31:

  1. Load cached single-person dual-CP rank-RANK_MAX features (amp + phase).
  2. Energy-sort each stream's components so a length-R prefix = the top-R
     components -> nested ranks (zero switching cost).
  3. Train ONE dual model with component-count dropout (random target rank per
     batch, tail components zeroed in BOTH streams) so it works at any prefix rank.
  4. Evaluate the single model at each rank in CANDIDATE_RANKS -> real A(R) in
     MPJPE (mm) plus per-frame MPJPE for the PRA contention sim (exp33).

Also dumps a per-frame mean-pose baseline MPJPE so the contention sim can charge
a missed-deadline frame the "no valid pose" cost.
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
sys.path.append(str(ROOT / "experiments"))

from itertools import permutations  # noqa: E402

from exp17_piw3d_cp_saff import (  # noqa: E402
    BASE_SUBCARRIERS,
    JOINTS,
    LINKS,
    MAX_PEOPLE,
    PACKETS,
    gate_entropy,
    matched_mpjpe_mm,
    pck_mm,
    query_set_loss,
    mean_pose_prediction,
    standardize,
    standardize_pose,
)
from exp19_piw3d_dualcp_saff import make_dual_model  # noqa: E402

RANK_MAX = 24
CANDIDATE_RANKS = [4, 8, 16, 24]
FEAT = LINKS + BASE_SUBCARRIERS + PACKETS  # 59 per stream


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required.") from exc
    return torch, nn, DataLoader, TensorDataset


def energy_sort_stream(x, rank):
    """Reorder one stream's CP components per frame by rank-1 energy (descending)."""
    n = x.shape[0]
    a_size = LINKS * rank
    b_size = BASE_SUBCARRIERS * rank
    a = x[:, :a_size].reshape(n, rank, LINKS)
    b = x[:, a_size : a_size + b_size].reshape(n, rank, BASE_SUBCARRIERS)
    c = x[:, a_size + b_size :].reshape(n, rank, PACKETS)
    energy = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) * np.linalg.norm(c, axis=2)
    order = np.argsort(-energy, axis=1)
    a = np.take_along_axis(a, order[:, :, None], axis=1)
    b = np.take_along_axis(b, order[:, :, None], axis=1)
    c = np.take_along_axis(c, order[:, :, None], axis=1)
    return np.concatenate([a.reshape(n, a_size), b.reshape(n, b_size), c.reshape(n, PACKETS * rank)], axis=1).astype(np.float32)


def mask_rank_flat(xb, rank, torch):
    """Zero tail components beyond `rank` for one stream (energy-sorted -> prefix = top-R)."""
    if rank >= RANK_MAX:
        return xb
    x = xb.clone()
    a_size = LINKS * RANK_MAX
    b_size = BASE_SUBCARRIERS * RANK_MAX
    x[:, rank * LINKS : a_size] = 0.0
    x[:, a_size + rank * BASE_SUBCARRIERS : a_size + b_size] = 0.0
    x[:, a_size + b_size + rank * PACKETS :] = 0.0
    return x


def per_frame_mpjpe_mm(y_true, y_pred, mask):
    """Per-frame matched MPJPE (mm); returns (values, valid_mask)."""
    perms = list(permutations(range(MAX_PEOPLE)))
    out = np.zeros(len(y_true), dtype=np.float32)
    valid = np.zeros(len(y_true), dtype=bool)
    for i, (yt, yp, m) in enumerate(zip(y_true, y_pred, mask)):
        n = int(m.sum())
        if n == 0:
            continue
        best = None
        for perm in perms:
            err = np.linalg.norm(yt[:n] - yp[list(perm)[:n]], axis=-1).mean()
            if best is None or err < best:
                best = err
        out[i] = best * 1000.0
        valid[i] = True
    return out, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs" / "piw3d_dualcp_1p_rank24_features.npz")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "piw_dualcp_anytime_ranks.csv")
    parser.add_argument("--dump", type=Path, default=ROOT / "outputs" / "piw_dualcp_anytime_rank_eval.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--query-mixer", choices=["none", "attention"], default="attention")
    parser.add_argument("--pose-loss", choices=["mse", "l1", "smooth_l1"], default="l1")
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--bone-weight", type=float, default=0.02)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank-sampling", choices=["uniform", "sandwich"], default="sandwich",
                        help="uniform: one random rank per batch (old, under-trains high ranks); "
                             "sandwich: train min+max+random-middle each batch (anytime-net standard)")
    parser.add_argument("--distill", action="store_true",
                        help="inplace KD: sub-ranks match the full-rank (teacher) pose output")
    parser.add_argument("--distill-weight", type=float, default=1.0,
                        help="weight on the KD (teacher-matching) term for sub-ranks")
    parser.add_argument("--distill-gt-weight", type=float, default=0.5,
                        help="weight on the ground-truth term for sub-ranks under KD")
    parser.add_argument("--distill-warmup", type=int, default=10,
                        help="epochs of plain GT training before KD kicks in (teacher must be usable)")
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(args.features, allow_pickle=True)
    xa_train = z["x_amp_train"].astype(np.float32)
    xp_train = z["x_phase_train"].astype(np.float32)
    xa_test = z["x_amp_test"].astype(np.float32)
    xp_test = z["x_phase_test"].astype(np.float32)
    y_train = z["y_train"].astype(np.float32)
    y_test = z["y_test"].astype(np.float32)
    mask_train = z["mask_train"].astype(np.float32)
    mask_test = z["mask_test"].astype(np.float32)

    rank_in = xa_train.shape[1] // FEAT
    if rank_in != RANK_MAX:
        raise SystemExit(f"Feature rank {rank_in} != RANK_MAX {RANK_MAX} (feat_dim={xa_train.shape[1]}, FEAT={FEAT}).")
    print(f"loaded: amp={xa_train.shape} phase={xp_train.shape} test={xa_test.shape} rank={rank_in}", flush=True)

    if args.max_train:
        xa_train, xp_train = xa_train[: args.max_train], xp_train[: args.max_train]
        y_train, mask_train = y_train[: args.max_train], mask_train[: args.max_train]
    if args.max_test:
        xa_test, xp_test = xa_test[: args.max_test], xp_test[: args.max_test]
        y_test, mask_test = y_test[: args.max_test], mask_test[: args.max_test]

    # Energy-sort each stream so a prefix = top-R components.
    xa_train = energy_sort_stream(xa_train, RANK_MAX)
    xp_train = energy_sort_stream(xp_train, RANK_MAX)
    xa_test = energy_sort_stream(xa_test, RANK_MAX)
    xp_test = energy_sort_stream(xp_test, RANK_MAX)

    # Match the dedicated dual-CP training protocol from exp19: standardize
    # both CP streams and train in standardized pose coordinates. The previous
    # anytime run optimized raw 3D coordinates and collapsed at full rank.
    xa_train, xa_test = standardize(xa_train, xa_test)
    xp_train, xp_test = standardize(xp_train, xp_test)
    y_train_s, _, y_mean, y_std = standardize_pose(y_train, y_test)

    device = torch.device(args.device)
    model = make_dual_model(nn, RANK_MAX, args.model_size, args.num_queries, args.query_mixer, args.gate_temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.from_numpy(xa_train), torch.from_numpy(xp_train),
        torch.from_numpy(y_train_s.astype(np.float32)), torch.from_numpy(mask_train),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    params = sum(p.numel() for p in model.parameters())
    print(f"model_params={params} candidate_ranks={CANDIDATE_RANKS}", flush=True)

    rng = np.random.RandomState(args.seed)
    lo, hi = min(CANDIDATE_RANKS), max(CANDIDATE_RANKS)
    mids = [r for r in CANDIDATE_RANKS if r not in (lo, hi)]
    print(f"rank_sampling={args.rank_sampling} lo={lo} hi={hi} mids={mids}", flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xa, xp, yb, mb in loader:
            xa, xp, yb, mb = xa.to(device), xp.to(device), yb.to(device), mb.to(device)
            if args.rank_sampling == "sandwich":
                # Anytime-net "sandwich rule": always train the smallest and largest
                # rank (bracket the operating range) plus one random middle rank.
                r_set = [lo, hi] + ([int(rng.choice(mids))] if mids else [])
            else:
                r_set = [int(rng.choice(CANDIDATE_RANKS))]
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            kd_on = args.distill and epoch > args.distill_warmup

            def gt_loss(poses, logits, count_logits, gates):
                loss = query_set_loss(poses, logits, count_logits, yb, mb, torch,
                                      args.cls_weight, args.count_weight, args.bone_weight, args.pose_loss)
                if args.gate_entropy_weight > 0:
                    loss = loss + args.gate_entropy_weight * (gate_entropy(gates[:, :4]) + gate_entropy(gates[:, 4:8]))
                return loss

            if kd_on:
                # Teacher = full (hi) rank, trained on ground truth. Sub-ranks match its output.
                xa_hi = mask_rank_flat(xa, hi, torch)
                xp_hi = mask_rank_flat(xp, hi, torch)
                poses_hi, logits_hi, count_hi, gates_hi = model(xa_hi, xp_hi, return_gates=True)
                loss_hi = gt_loss(poses_hi, logits_hi, count_hi, gates_hi)
                loss_hi.backward()
                batch_loss += float(loss_hi.detach().cpu())
                teacher_poses = poses_hi.detach()
                for r in [rr for rr in r_set if rr != hi]:
                    xa_r = mask_rank_flat(xa, r, torch)
                    xp_r = mask_rank_flat(xp, r, torch)
                    poses_r, logits_r, count_r, gates_r = model(xa_r, xp_r, return_gates=True)
                    kd = torch.nn.functional.l1_loss(poses_r, teacher_poses)
                    loss_r = args.distill_weight * kd + args.distill_gt_weight * gt_loss(
                        poses_r, logits_r, count_r, gates_r)
                    loss_r.backward()
                    batch_loss += float(loss_r.detach().cpu())
            else:
                for r in r_set:
                    xa_r = mask_rank_flat(xa, r, torch)
                    xp_r = mask_rank_flat(xp, r, torch)
                    poses, logits, count_logits, gates = model(xa_r, xp_r, return_gates=True)
                    loss = gt_loss(poses, logits, count_logits, gates) / len(r_set)
                    loss.backward()
                    batch_loss += float(loss.detach().cpu())
            opt.step()
            losses.append(batch_loss)
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0

    # Per-frame mean-pose baseline (the cost charged for a missed-deadline frame).
    mean_pred = mean_pose_prediction(y_train, mask_train, len(y_test))
    mean_pf, valid = per_frame_mpjpe_mm(y_test, mean_pred, mask_test)

    model.eval()
    rows = []
    mpjpe_by_rank = np.zeros((len(y_test), len(CANDIDATE_RANKS)), dtype=np.float32)
    xa_test_t = torch.from_numpy(xa_test)
    xp_test_t = torch.from_numpy(xp_test)
    for j, r in enumerate(CANDIDATE_RANKS):
        preds, scores = [], []
        t1 = time.time()
        with torch.no_grad():
            for start in range(0, len(xa_test), args.batch_size):
                xa = mask_rank_flat(xa_test_t[start : start + args.batch_size].to(device), r, torch)
                xp = mask_rank_flat(xp_test_t[start : start + args.batch_size].to(device), r, torch)
                poses, logits, _, _ = model(xa, xp, return_gates=True)
                preds.append(poses.cpu().numpy())
                scores.append(torch.sigmoid(logits).cpu().numpy())
        pred_sec = time.time() - t1
        pred = np.vstack(preds)
        score = np.vstack(scores)
        top = np.argsort(-score, axis=1)[:, :MAX_PEOPLE]
        pred_top = np.take_along_axis(pred, top[:, :, None, None], axis=1)
        pred_top = pred_top * y_std + y_mean
        pf, _ = per_frame_mpjpe_mm(y_test, pred_top, mask_test)
        mpjpe_by_rank[:, j] = pf
        row = {
            "rank": r,
            "mpjpe_mm": matched_mpjpe_mm(y_test, pred_top, mask_test),
            "pck50": pck_mm(y_test, pred_top, mask_test, 50.0),
            "pck100": pck_mm(y_test, pred_top, mask_test, 100.0),
            "params": params,
            "predict_us_per_sample": 1e6 * pred_sec / len(y_test),
        }
        rows.append(row)
        print(row, flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    keep = valid
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dump,
        ranks=np.array(CANDIDATE_RANKS),
        mpjpe_by_rank_mm=mpjpe_by_rank[keep],       # per-frame MPJPE at each rank (mm)
        mean_pose_by_frame_mm=mean_pf[keep],         # per-frame mean-pose baseline MPJPE (mm)
        A_R_mpjpe_mm=np.array([r["mpjpe_mm"] for r in rows], dtype=np.float32),
        params=params,
        train_sec=train_sec,
    )
    summary = {
        "train_sec": train_sec,
        "params": int(params),
        "A_R_mpjpe_mm": {r["rank"]: r["mpjpe_mm"] for r in rows},
        "A_R_pck100": {r["rank"]: r["pck100"] for r in rows},
        "mean_pose_mpjpe_mm": float(mean_pf[keep].mean()),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    print(f"saved_csv={args.csv}")
    print(f"saved_dump={args.dump}")


if __name__ == "__main__":
    main()
