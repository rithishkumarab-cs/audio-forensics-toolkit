"""Plotting helpers used by the dashboard and PDF report."""
from __future__ import annotations

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def waveform_figure(signal: np.ndarray, sample_rate: int, max_points: int = 120_000):
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    stride = max(1, int(np.ceil(len(x) / max_points)))
    y = x[::stride]
    t = np.arange(len(y)) * stride / sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    ax.plot(t, y, linewidth=0.55, color="#3568A8")
    ax.set_title("Audio waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    return fig


def spectrogram_figure(signal: np.ndarray, sample_rate: int, max_seconds: float = 60.0):
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    x = x[: int(max_seconds * sample_rate)]
    f, t, sxx = spectrogram(x, fs=sample_rate, nperseg=512, noverlap=384, mode="magnitude")
    db = 20 * np.log10(np.maximum(sxx, 1e-9))
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    mesh = ax.pcolormesh(t, f, db, shading="auto", cmap="magma")
    ax.set_ylim(0, min(8000, sample_rate / 2))
    ax.set_title("Spectrogram (first 60 seconds max)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    return fig


def figure_png_bytes(fig) -> bytes:
    buff = io.BytesIO()
    fig.savefig(buff, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buff.seek(0)
    return buff.read()
