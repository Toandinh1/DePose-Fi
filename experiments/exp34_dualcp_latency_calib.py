"""Calibrate real per-rank dual-CP extraction latency on PiW-sized tensors.

Times nonnegative_cp_mu at each candidate rank on a (LINKS, BASE_SUBCARRIERS,
PACKETS) = (9, 30, 20) tensor (one CSI stream), doubles it for dual-CP (amp +
phase), adds a fixed S-AFF term, and writes a latency-json mapping rank -> total
per-frame microseconds for exp33. This removes the guessed latency anchor.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from cp_factorization import nonnegative_cp_mu  # noqa: E402

LINKS, SUB, PACK = 9, 30, 20
RANKS = [4, 8, 16, 24]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dualcp_latency_us.json")
    parser.add_argument("--cp-iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--saff-us", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    L = {}
    single_cp = {}
    for r in RANKS:
        times = []
        for k in range(args.repeats + args.warmup):
            x = np.abs(rng.randn(LINKS, SUB, PACK)).astype(np.float64)
            t0 = time.perf_counter()
            nonnegative_cp_mu(x, rank=r, iters=args.cp_iters, seed=k)
            dt = (time.perf_counter() - t0) * 1e6  # us
            if k >= args.warmup:
                times.append(dt)
        cp_us = float(np.median(times))
        single_cp[r] = cp_us
        L[r] = 2.0 * cp_us + args.saff_us  # dual-CP (amp + phase) + S-AFF
        print(f"rank={r}: single_cp_us={cp_us:.1f} dual+saff_us={L[r]:.1f} (n={len(times)})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({str(r): round(v, 1) for r, v in L.items()}, indent=2))
    print("single_cp_us=" + json.dumps({r: round(v, 1) for r, v in single_cp.items()}))
    print("L(R)_ms=" + json.dumps({r: round(v / 1000.0, 3) for r, v in L.items()}))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
