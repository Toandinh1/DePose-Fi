from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import cp_reconstruct, nonnegative_cp_mu  # noqa: E402


def main():
    out_dir = ROOT / "outputs"
    data = np.load(out_dir / "E01_E01_S01_A01_ldt_120frames.npz")
    x = data["X_norm"]

    ranks = list(range(1, 11))
    errors = []
    temporal_tv = []
    for rank in ranks:
        a, b, c, _ = nonnegative_cp_mu(x, rank=rank, iters=500, seed=rank)
        recon = cp_reconstruct(a, b, c)
        err = np.linalg.norm(x - recon) / np.linalg.norm(x)
        tv = np.mean(np.abs(np.diff(c, axis=0)))
        errors.append(err)
        temporal_tv.append(tv)
        print(f"rank={rank} rel_err={err:.4f} mean_temporal_tv={tv:.4f}")

    out_npz = out_dir / "E01_E01_S01_A01_cp_rank_sweep.npz"
    np.savez_compressed(out_npz, ranks=np.asarray(ranks), errors=np.asarray(errors), temporal_tv=np.asarray(temporal_tv))

    fig, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax1.plot(ranks, errors, marker="o", label="reconstruction error")
    ax1.set_xlabel("CP rank R")
    ax1.set_ylabel("relative reconstruction error")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ranks, temporal_tv, marker="s", color="tab:orange", label="temporal variation")
    ax2.set_ylabel("mean |Delta C|")
    fig.suptitle("CP rank sweep on MM-Fi sample")
    out_png = out_dir / "E01_E01_S01_A01_cp_rank_sweep.png"
    fig.savefig(out_png, dpi=180)
    print(f"saved_npz={out_npz}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
