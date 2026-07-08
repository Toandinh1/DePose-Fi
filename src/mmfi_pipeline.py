from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import stft


def load_wifi_csi_frames(frame_dir, limit=None):
    """Load MM-Fi wifi-csi frame*.mat files into H with shape L x F x T."""
    frame_dir = Path(frame_dir)
    paths = sorted(frame_dir.glob("frame*.mat"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No frame*.mat files found in {frame_dir}")

    chunks = []
    for path in paths:
        mat = loadmat(path)
        amp = np.asarray(mat["CSIamp"], dtype=np.float64)
        phase = np.asarray(mat["CSIphase"], dtype=np.float64)
        if amp.shape != phase.shape:
            raise ValueError(f"Shape mismatch in {path}: {amp.shape} vs {phase.shape}")
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        # MM-Fi sample frame shape: links x subcarriers x short-time packets.
        chunks.append(amp * np.exp(1j * phase))
    return np.concatenate(chunks, axis=2), paths


def remove_static_component(h):
    """Subtract temporal mean from complex CSI H with shape L x F x T."""
    static = h.mean(axis=2, keepdims=True)
    return h - static, static


def build_link_doppler_time_tensor(delta_h, nperseg=64, noverlap=48, nfft=64):
    """Build X(l, d, tau) = sum_f |STFT_t(delta H(l,f,t))|^2."""
    links, subcarriers, _ = delta_h.shape
    tensors = []
    freqs_ref = None
    times_ref = None
    for link in range(links):
        power_accum = None
        for sub in range(subcarriers):
            freqs, times, z = stft(
                delta_h[link, sub, :],
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=False,
                boundary=None,
                padded=False,
            )
            power = np.abs(np.fft.fftshift(z, axes=0)) ** 2
            if power_accum is None:
                power_accum = power
                freqs_ref = np.fft.fftshift(freqs)
                times_ref = times
            else:
                power_accum += power
        tensors.append(power_accum)
    x = np.stack(tensors, axis=0)
    return x.real.astype(np.float32), freqs_ref, times_ref


def normalize_tensor(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    scale = np.percentile(x, 99.0) + eps
    return np.clip(x / scale, 0.0, None)
