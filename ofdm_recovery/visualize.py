"""Visualization helpers for the OFDM subframe generator.

Three views of the same subframe:
    1. Time-domain I/Q (clean vs noisy)
    2. Spectrogram (STFT magnitude in dB)
    3. Post-FFT constellation (scatter of active subcarriers across symbols)

STFT is hand-rolled to avoid adding scipy as a dependency.
"""

import numpy as np
import matplotlib.pyplot as plt

from lte_params import FS, N_FFT, active_subcarrier_indices
from ofdm_signal import strip_cp_and_fft


def _sliding_stft(x, nperseg=256, noverlap=192, window="hann"):
    """Hand-rolled STFT returning (freqs_Hz, times_s, Sxx_complex).

    Args:
        x: 1-D complex baseband signal.
        nperseg: samples per segment.
        noverlap: overlapping samples between segments.
        window: either "hann" or "rect".

    Returns:
        freqs:     np.ndarray of shape (nperseg,), shifted to [-Fs/2, +Fs/2).
        times:     np.ndarray of shape (n_segments,) — segment center times (s).
        Sxx:       complex np.ndarray of shape (nperseg, n_segments),
                   already fft-shifted along axis 0.
    """
    hop = nperseg - noverlap
    n = len(x)
    n_segments = 1 + (n - nperseg) // hop

    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    else:
        w = np.ones(nperseg)

    Sxx = np.empty((nperseg, n_segments), dtype=np.complex128)
    for i in range(n_segments):
        start = i * hop
        seg = x[start:start + nperseg] * w
        Sxx[:, i] = np.fft.fftshift(np.fft.fft(seg))

    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / FS))
    times = (np.arange(n_segments) * hop + nperseg / 2) / FS
    return freqs, times, Sxx


def plot_time_domain(rx, clean, save_path, n_show=512):
    """I and Q over the first `n_show` samples, clean and noisy overlaid."""
    t = np.arange(n_show) / FS * 1e6   # microseconds

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax_i.plot(t, clean[:n_show].real, label="clean", linewidth=1.0)
    ax_i.plot(t, rx[:n_show].real, label="received (noisy)",
              alpha=0.6, linewidth=0.8)
    ax_i.set_ylabel("In-phase")
    ax_i.legend(loc="upper right")
    ax_i.grid(alpha=0.3)

    ax_q.plot(t, clean[:n_show].imag, label="clean", linewidth=1.0)
    ax_q.plot(t, rx[:n_show].imag, label="received (noisy)",
              alpha=0.6, linewidth=0.8)
    ax_q.set_ylabel("Quadrature")
    ax_q.set_xlabel("Time (us)")
    ax_q.legend(loc="upper right")
    ax_q.grid(alpha=0.3)

    fig.suptitle(f"LTE-5 MHz subframe: first {n_show} samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_spectrogram(rx, save_path, nperseg=256, noverlap=192):
    """STFT magnitude (dB) of the full subframe, freq in MHz, time in ms."""
    freqs, times, Sxx = _sliding_stft(rx, nperseg=nperseg, noverlap=noverlap)
    mag_db = 20 * np.log10(np.abs(Sxx) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(
        times * 1e3, freqs * 1e-6, mag_db,
        shading="auto", cmap="viridis",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(
        f"Spectrogram  (nperseg={nperseg}, noverlap={noverlap}, Fs={FS*1e-6:.2f} MHz)"
    )
    # Mark the 5 MHz occupied band edges (roughly +/- 2.25 MHz for 300 x 15 kHz)
    for edge in (-2.25, 2.25):
        ax.axhline(edge, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_constellation(rx, save_path):
    """Scatter all active-subcarrier symbols from the received subframe.

    Strips CP, 512-pt FFT each symbol, scatters the 300 * 14 = 4200 points
    on a complex plane.
    """
    rx_freq = strip_cp_and_fft(rx)
    active = np.asarray(active_subcarrier_indices())
    points = rx_freq[:, active].reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points.real, points.imag, s=3, alpha=0.35, color="C0")

    # Reference QPSK locations
    ideal = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    ax.scatter(ideal.real, ideal.imag, s=80, facecolor="none",
               edgecolor="red", linewidth=1.5, label="ideal QPSK")
    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(f"Received constellation: {len(points)} active-subcarrier symbols")
    ax.set_aspect("equal")

    # Tight, but leave room for outliers at low SNR
    lim = max(1.2, 1.2 * np.percentile(np.abs(points), 99))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
