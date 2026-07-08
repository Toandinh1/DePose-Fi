from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import cp_reconstruct, nonnegative_cp_mu  # noqa: E402


def main():
    out_dir = ROOT / "outputs"
    tensor_path = out_dir / "E01_E01_S01_A01_ldt_120frames.npz"
    data = np.load(tensor_path)
    x = data["X_norm"]

    rank = 4
    a, b, c, history = nonnegative_cp_mu(x, rank=rank, iters=600, seed=2)
    recon = cp_reconstruct(a, b, c)
    rel_err = np.linalg.norm(x - recon) / np.linalg.norm(x)

    out_npz = out_dir / "E01_E01_S01_A01_cp_rank4.npz"
    np.savez_compressed(out_npz, A=a, B=b, C=c, recon=recon, history=history, rel_err=rel_err)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    im0 = axes[0, 0].imshow(a, aspect="auto")
    axes[0, 0].set_title("Link factors A")
    axes[0, 0].set_xlabel("component")
    axes[0, 0].set_ylabel("link")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(b, aspect="auto", origin="lower")
    axes[0, 1].set_title("Doppler factors B")
    axes[0, 1].set_xlabel("component")
    axes[0, 1].set_ylabel("Doppler bin")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].plot(c)
    axes[1, 0].set_title("Temporal activations C")
    axes[1, 0].set_xlabel("STFT time window")

    axes[1, 1].plot(history, marker="o")
    axes[1, 1].set_title("CP relative reconstruction error")
    axes[1, 1].set_xlabel("checkpoint")
    axes[1, 1].set_ylabel("relative error")

    out_png = out_dir / "E01_E01_S01_A01_cp_rank4.png"
    fig.savefig(out_png, dpi=180)

    print(f"input_shape={x.shape}")
    print(f"rank={rank}")
    print(f"relative_reconstruction_error={rel_err:.4f}")
    print(f"A_shape={a.shape} B_shape={b.shape} C_shape={c.shape}")
    print(f"saved_npz={out_npz}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
