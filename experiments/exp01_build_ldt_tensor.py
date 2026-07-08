from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mmfi_pipeline import (  # noqa: E402
    build_link_doppler_time_tensor,
    load_wifi_csi_frames,
    normalize_tensor,
    remove_static_component,
)


def main():
    frame_dir = ROOT / "data" / "MMFi_sample" / "E01_E01_S01_A01_wifi"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    h, paths = load_wifi_csi_frames(frame_dir, limit=120)
    delta_h, static = remove_static_component(h)
    x, doppler_bins, time_bins = build_link_doppler_time_tensor(
        delta_h, nperseg=64, noverlap=48, nfft=64
    )
    x_norm = normalize_tensor(x)

    out_npz = out_dir / "E01_E01_S01_A01_ldt_120frames.npz"
    np.savez_compressed(
        out_npz,
        X=x,
        X_norm=x_norm,
        doppler_bins=doppler_bins,
        time_bins=time_bins,
        static_abs=np.abs(static).astype(np.float32),
        frame_count=len(paths),
        source_dir=str(frame_dir),
    )

    link_energy = x.sum(axis=(1, 2))
    doppler_time = x.sum(axis=0)
    dynamic_energy = np.abs(delta_h).mean(axis=(0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(dynamic_energy)
    axes[0].set_title("Dynamic CSI energy")
    axes[0].set_xlabel("CSI packet index")
    axes[0].set_ylabel("mean |Delta H|")

    axes[1].bar(np.arange(len(link_energy)), link_energy)
    axes[1].set_title("Link energy")
    axes[1].set_xlabel("link")

    im = axes[2].imshow(doppler_time, aspect="auto", origin="lower")
    axes[2].set_title("Aggregated Doppler-time tensor")
    axes[2].set_xlabel("STFT time window")
    axes[2].set_ylabel("Doppler bin")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    out_png = out_dir / "E01_E01_S01_A01_ldt_120frames.png"
    fig.savefig(out_png, dpi=180)

    print(f"frames_loaded={len(paths)}")
    print(f"H_shape={h.shape}")
    print(f"DeltaH_shape={delta_h.shape}")
    print(f"LDT_shape={x.shape}")
    print(f"LDT_min={x.min():.6g} LDT_max={x.max():.6g} LDT_mean={x.mean():.6g}")
    print(f"saved_npz={out_npz}")
    print(f"saved_png={out_png}")


if __name__ == "__main__":
    main()
