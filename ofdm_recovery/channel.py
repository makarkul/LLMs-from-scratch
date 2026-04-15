"""Multipath fading channel models.

Currently: 3GPP EPA (Extended Pedestrian A) with block-fading Rayleigh taps.
We synthesize the channel directly in the frequency domain, which lets each
tap have a fractional-sample delay without any interpolation filtering —
    H[k] = sum_l  alpha_l * exp(-j 2 pi k tau_l / N_FFT)
Since the CP absorbs all multipath (EPA max delay 410 ns = ~3.15 samples at
Fs = 7.68 MHz, CP >= 36 samples), applying H per subcarrier is equivalent to
time-domain linear convolution followed by CP strip.
"""

import numpy as np

from lte_params import FS, N_FFT

# 3GPP TS 36.101 Annex B.2 — Extended Pedestrian A (EPA)
EPA_DELAYS_NS = np.array([0.0, 30.0, 70.0, 90.0, 110.0, 190.0, 410.0])
EPA_POWERS_DB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])


def _power_profile(powers_db):
    """Linear power per tap, normalized so the total is unity."""
    p = 10.0 ** (powers_db / 10.0)
    return p / p.sum()


def epa_frequency_response(rng=None, n_fft=N_FFT, fs=FS):
    """Sample one block-faded EPA channel realization as a freq response.

    Args:
        rng: numpy Generator. If None, a fresh default_rng() is used.
        n_fft: FFT size (default N_FFT).
        fs: sample rate (default FS).

    Returns:
        H: complex array of shape (n_fft,). E[|H[k]|^2] = 1 by construction.
    """
    if rng is None:
        rng = np.random.default_rng()

    delays_samples = EPA_DELAYS_NS * 1e-9 * fs           # fractional samples
    powers_lin = _power_profile(EPA_POWERS_DB)

    # Rayleigh taps: complex Gaussian, per-tap variance = power_l
    n_taps = len(delays_samples)
    alphas = (
        rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)
    ) * np.sqrt(powers_lin / 2.0)

    k = np.arange(n_fft)
    H = np.zeros(n_fft, dtype=np.complex128)
    for a, tau in zip(alphas, delays_samples):
        H += a * np.exp(-1j * 2.0 * np.pi * k * tau / n_fft)
    return H


def sample_channel(profile="awgn", rng=None):
    """Dispatch to the named channel profile.

    Returns the freq response H of shape (N_FFT,), or None for AWGN.
    """
    if profile == "awgn":
        return None
    if profile == "epa":
        return epa_frequency_response(rng=rng)
    raise ValueError(f"unknown channel profile: {profile!r}")
