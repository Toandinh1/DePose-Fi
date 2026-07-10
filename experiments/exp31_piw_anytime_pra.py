"""Anytime CP + query-S-AFF for Proactive Rank Adaptation (PRA) on Person-in-WiFi-3D.

Motivation
----------
On MM-Fi the accuracy-vs-rank curve A(R) is nearly flat (rank 2->8 buys ~1.5
PCK20), so rank adaptation has almost nothing to optimise and a fixed cheap rank
wins. The multi-person hypothesis: PiW needs more rank-1 components to represent
several people, so A(R) should be *steep*, giving PRA real headroom.

This experiment mirrors exp29 (the MM-Fi anytime model) but for PiW's multi-person
query set-prediction head (reused from exp17):

  1. Load cached rank-RANK_MAX PiW CP features (CP extraction already done).
  2. Energy-sort each frame's components (||a_r|| * ||b_r|| * ||c_r|| desc) so a
     length-R prefix = the top-R components -> nested / zero-switching-cost ranks.
  3. Train ONE query-S-AFF with component-count dropout (random target rank per
     batch, tail components zeroed) so it is robust at any prefix rank.
  4. Evaluate the single model at each rank in CANDIDATE_RANKS, producing a real
     A(R) and a per-frame PCK-by-rank array for the PRA contention sim (exp30).

Only CP extraction cost scales with R; the head sees zero-padded tail components,
so parameter count is constant.
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

# Reuse the PiW query head, losses and matching from exp17.
import exp17_piw3d_cp_saff as piw  # noqa: E402
from exp17_piw3d_cp_saff import (  # noqa: E402
    LINKS,
    PACKETS,
    JOINTS,
    MAX_PEOPLE,
    make_query_saff,
    query_set_loss,
    gate_entropy,
    infer_subcarriers,
    matched_mpjpe_mm,
    pck_mm,
    summarize_gates,
)

RANK_MAX = 16
CANDIDATE_RANKS = [2, 4, 8, 12, 16]
PF_PCK_MM = 100.0  # per-frame PCK threshold (mm) used as the PRA accuracy signal


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required.") from exc
    return torch, nn, DataLoader, TensorDataset


def energy_sort_components(x, rank, subcarriers):
    """Reorder CP components per frame by rank-1 energy (descending).

    x layout (matches exp17): [a: LINKS*rank | b: subcarriers*rank | c: PACKETS*rank],
    each block component-major so component r is contiguous within its block.
    """
    n = x.shape[0]
    a_size = LINKS * rank
    b_size = subcarriers * rank
    a = x[:, :a_size].reshape(n, rank, LINKS)
    b = x[:, a_size : a_size + b_size].reshape(n, rank, subcarriers)
    c = x[:, a_size + b_size :].reshape(n, rank, PACKETS)
    energy = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) * np.linalg.norm(c, axis=2)
    order = np.argsort(-energy, axis=1)  # N x rank, descending
    a = np.take_along_axis(a, order[:, :, None], axis=1)
    b = np.take_along_axis(b, order[:, :, None], axis=1)
    c = np.take_along_axis(c, order[:, :, None], axis=1)
    return np.concatenate(
        [a.reshape(n, a_size), b.reshape(n, b_size), c.reshape(n, PACKETS * rank)], axis=1
    ).astype(np.float32)


def mask_rank_flat(xb, rank, subcarriers, torch):
    """Zero tail components beyond `rank` (energy-sorted, so prefix = top-R)."""
    if rank >= RANK_MAX:
        return xb
    x = xb.clone()
    a_size = LINKS * RANK_MAX
    b_size = subcarriers * RANK_MAX
    x[:, rank * LINKS : a_size] = 0.0
    x[:, a_size + rank * subcarriers : a_size + b_size] = 0.0
    x[:, a_size + b_size + rank * PACKETS :] = 0.0
    return x


def per_frame_pck_mm(y_true, y_pred, mask, threshold_mm):
    """Per-frame PCK: fraction of that frame's valid (person,joint) within threshold.

    Uses the same best-permutation matching as exp17.pck_mm, but returns one value
    per frame (0 if no valid people, marked via nan-free 0 and excluded via mask)."""
    from itertools import permutations

    perms = list(permutations(range(MAX_PEOPLE)))
    threshold = threshold_mm / 1000.0
    out = np.zeros(len(y_true), dtype=np.float32)
    valid = np.zeros(len(y_true), dtype=bool)
    for i, (yt, yp, m) in enumerate(zip(y_true, y_pred, mask)):
        n = int(m.sum())
        if n == 0:
            continue
        best_err = None
        for perm in perms:
            pred_sel = yp[list(perm)[:n]]
            err = np.linalg.norm(yt[:n] - pred_sel, axis=-1)
            if best_err is None or err.mean() < best_err.mean():
                best_err = err
        out[i] = 100.0 * (best_err <= threshold).mean()
        valid[i] = True
    return out, valid


def per_frame_mpjpe_mm(y_true, y_pred, mask):
    """Per-frame best-permutation MPJPE (mm) and per-frame person count.

    Same matching as exp17.matched_mpjpe_mm but returns one value per frame so the
    contention sim (exp33) can score delivered quality per slot, plus the number of
    valid people in each frame for person-count stratification."""
    from itertools import permutations

    perms = list(permutations(range(MAX_PEOPLE)))
    out = np.zeros(len(y_true), dtype=np.float32)
    counts = np.zeros(len(y_true), dtype=np.int32)
    valid = np.zeros(len(y_true), dtype=bool)
    for i, (yt, yp, m) in enumerate(zip(y_true, y_pred, mask)):
        n = int(m.sum())
        if n == 0:
            continue
        best = None
        for perm in perms:
            pred_sel = yp[list(perm)[:n]]
            err = np.linalg.norm(yt[:n] - pred_sel, axis=-1).mean()
            if best is None or err < best:
                best = err
        out[i] = best * 1000.0
        counts[i] = n
        valid[i] = True
    return out, counts, valid


def mean_pose_baseline_mm(y_true, mask, mean_pose):
    """Per-frame MPJPE (mm) if the trivial mean training pose were emitted.

    Used as the 'no valid pose delivered this slot' penalty for missed deadlines in
    the contention sim (scoring a miss as 0 error would be nonsensical for MPJPE)."""
    out = np.zeros(len(y_true), dtype=np.float32)
    for i, (yt, m) in enumerate(zip(y_true, mask)):
        n = int(m.sum())
        if n == 0:
            continue
        err = np.linalg.norm(yt[:n] - mean_pose[None], axis=-1).mean()
        out[i] = err * 1000.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs" / "piw3d_full_sanitized_phase_rank16_features.npz")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "piw_anytime_ranks.csv")
    parser.add_argument("--dump", type=Path, default=ROOT / "outputs" / "piw_anytime_pra_rank_eval.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--query-mixer", choices=["none", "gru", "attention"], default="attention")
    parser.add_argument("--pose-loss", choices=["mse", "l1", "smooth_l1"], default="l1")
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--bone-weight", type=float, default=0.0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(args.features, allow_pickle=True)
    x_train = z["x_cp_train"].astype(np.float32)
    x_test = z["x_cp_test"].astype(np.float32)
    y_train = z["y_train"].astype(np.float32)
    y_test = z["y_test"].astype(np.float32)
    mask_train = z["mask_train"].astype(np.float32)
    mask_test = z["mask_test"].astype(np.float32)

    subcarriers = infer_subcarriers(x_train.shape[1], RANK_MAX)
    rank_in = x_train.shape[1] // (LINKS + subcarriers + PACKETS)
    if rank_in != RANK_MAX:
        raise SystemExit(f"Feature rank {rank_in} != RANK_MAX {RANK_MAX}. Regenerate features at rank {RANK_MAX}.")
    print(f"loaded features: train={x_train.shape} test={x_test.shape} subcarriers={subcarriers}", flush=True)

    if args.max_train:
        x_train, y_train, mask_train = x_train[: args.max_train], y_train[: args.max_train], mask_train[: args.max_train]
    if args.max_test:
        x_test, y_test, mask_test = x_test[: args.max_test], y_test[: args.max_test], mask_test[: args.max_test]

    # Energy-sort components so a prefix = the top-R components (nested ranks).
    x_train = energy_sort_components(x_train, RANK_MAX, subcarriers)
    x_test = energy_sort_components(x_test, RANK_MAX, subcarriers)

    device = torch.device(args.device)
    model = make_query_saff(
        nn, RANK_MAX, args.gate_temperature, subcarriers, args.model_size, args.num_queries, args.query_mixer
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(mask_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    params = sum(p.numel() for p in model.parameters())
    print(f"model_params={params} candidate_ranks={CANDIDATE_RANKS}", flush=True)

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            r = int(rng.choice(CANDIDATE_RANKS))  # component-count dropout
            xb_r = mask_rank_flat(xb, r, subcarriers, torch)
            opt.zero_grad(set_to_none=True)
            poses, logits, count_logits, gates = model(xb_r, return_gates=True)
            loss = query_set_loss(poses, logits, count_logits, yb, mb, torch,
                                  args.cls_weight, args.count_weight, args.bone_weight, args.pose_loss)
            if args.gate_entropy_weight > 0:
                loss = loss + args.gate_entropy_weight * gate_entropy(gates)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    train_sec = time.time() - t0

    # Evaluate the single model at each rank prefix.
    model.eval()
    rows = []
    pck_by_rank = np.zeros((len(y_test), len(CANDIDATE_RANKS)), dtype=np.float32)
    mpjpe_by_rank = np.zeros((len(y_test), len(CANDIDATE_RANKS)), dtype=np.float32)
    people_by_frame = None
    valid_frames = None
    x_test_t = torch.from_numpy(x_test)
    for j, r in enumerate(CANDIDATE_RANKS):
        preds, scores = [], []
        t1 = time.time()
        with torch.no_grad():
            for start in range(0, len(x_test), args.batch_size):
                xb = mask_rank_flat(x_test_t[start : start + args.batch_size].to(device), r, subcarriers, torch)
                poses, logits, _, _ = model(xb, return_gates=True)
                preds.append(poses.cpu().numpy())
                scores.append(torch.sigmoid(logits).cpu().numpy())
        pred_sec = time.time() - t1
        pred = np.vstack(preds)
        score = np.vstack(scores)
        top = np.argsort(-score, axis=1)[:, :MAX_PEOPLE]
        pred_top = np.take_along_axis(pred, top[:, :, None, None], axis=1)

        pf, valid = per_frame_pck_mm(y_test, pred_top, mask_test, PF_PCK_MM)
        pck_by_rank[:, j] = pf
        pf_err, counts, _ = per_frame_mpjpe_mm(y_test, pred_top, mask_test)
        mpjpe_by_rank[:, j] = pf_err
        people_by_frame = counts
        valid_frames = valid
        row = {
            "rank": r,
            "mpjpe_mm": matched_mpjpe_mm(y_test, pred_top, mask_test),
            f"pck{int(PF_PCK_MM)}": pck_mm(y_test, pred_top, mask_test, PF_PCK_MM),
            "pck50": pck_mm(y_test, pred_top, mask_test, 50.0),
            "pck150": pck_mm(y_test, pred_top, mask_test, 150.0),
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

    # Mean training pose -> per-frame "no pose delivered" baseline for missed deadlines.
    w = mask_train.reshape(-1)
    poses_flat = y_train.reshape(-1, JOINTS, 3)
    mean_pose = (poses_flat * w[:, None, None]).sum(0) / max(w.sum(), 1.0)
    base_mm = mean_pose_baseline_mm(y_test, mask_test, mean_pose)

    # Dump per-frame accuracy over VALID frames only.
    keep = valid_frames
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dump,
        ranks=np.array(CANDIDATE_RANKS),
        pck20_by_rank=pck_by_rank[keep],          # per-frame PCK@PF_PCK_MM (name kept for exp30 reuse)
        mpjpe_by_rank_mm=mpjpe_by_rank[keep],     # per-frame MPJPE (mm) by rank -> exp33 accuracy signal
        mean_pose_by_frame_mm=base_mm[keep],      # per-frame miss penalty (mm)
        people_by_frame=people_by_frame[keep],    # valid people per frame -> stratification
        pf_pck_threshold_mm=np.array(PF_PCK_MM),
        A_R_pck=np.array([r[f"pck{int(PF_PCK_MM)}"] for r in rows], dtype=np.float32),
        A_R_mpjpe_mm=np.array([r["mpjpe_mm"] for r in rows], dtype=np.float32),
        params=params,
        train_sec=train_sec,
    )

    # Person-count-stratified A(R) in MPJPE (mm): does steepness live in multi-person frames?
    strat_csv = args.csv.with_name(args.csv.stem + "_by_people.csv")
    with strat_csv.open("w", newline="", encoding="utf-8") as f:
        w2 = csv.writer(f)
        w2.writerow(["people", "n_frames"] + [f"mpjpe_r{r}" for r in CANDIDATE_RANKS]
                    + [f"delta_r{CANDIDATE_RANKS[0]}_to_r{CANDIDATE_RANKS[-1]}"])
        pc = people_by_frame[keep]
        mbr = mpjpe_by_rank[keep]
        for people in sorted(set(int(v) for v in pc if v > 0)):
            sel = pc == people
            means = [float(mbr[sel, j].mean()) for j in range(len(CANDIDATE_RANKS))]
            w2.writerow([people, int(sel.sum())] + [round(m, 2) for m in means]
                        + [round(means[0] - means[-1], 2)])
        means_all = [float(mbr[:, j].mean()) for j in range(len(CANDIDATE_RANKS))]
        w2.writerow(["all", int(len(pc))] + [round(m, 2) for m in means_all]
                    + [round(means_all[0] - means_all[-1], 2)])
    print(f"saved_stratified={strat_csv}")
    summary = {
        "train_sec": train_sec,
        "params": int(params),
        f"A_R_pck{int(PF_PCK_MM)}": {r["rank"]: r[f"pck{int(PF_PCK_MM)}"] for r in rows},
        "A_R_mpjpe_mm": {r["rank"]: r["mpjpe_mm"] for r in rows},
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    print(f"saved_csv={args.csv}")
    print(f"saved_dump={args.dump}")


if __name__ == "__main__":
    main()
