"""LTE-5 MHz OFDM subframe generator (QPSK data, AWGN channel).

Generates one subframe of random QPSK data on all 300 active subcarriers and
adds complex AWGN at a configurable post-FFT per-subcarrier SNR. No pilots,
no control channels — just data-carrying OFDM for the Option C experiments.
"""

import numpy as np

from lte_params import (
    N_FFT,
    N_ACTIVE,
    N_SYMBOLS_PER_SUBFRAME,
    SAMPLES_PER_SUBFRAME,
    cp_length,
    active_subcarrier_indices,
)


def qpsk_symbols(bits, rng=None):
    """Map bits -> unit-magnitude QPSK constellation points.

    Args:
        bits: array of shape (..., 2) or flat even-length; last dim pairs
              2 bits into one QPSK symbol with Gray coding:
                  (0,0) -> +1 +1j      (0,1) -> +1 -1j
                  (1,0) -> -1 +1j      (1,1) -> -1 -1j
              (scaled by 1/sqrt(2) for unit average power).

    Returns:
        complex array with shape bits.shape[:-1] if bits has a trailing pair dim,
        otherwise shape (len(bits)//2,).
    """
    bits = np.asarray(bits).astype(np.int8)
    if bits.shape[-1] != 2:
        assert bits.size % 2 == 0
        bits = bits.reshape(-1, 2)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    re = np.where(bits[..., 0] == 0, 1.0, -1.0)
    im = np.where(bits[..., 1] == 0, 1.0, -1.0)
    return (re + 1j * im) * inv_sqrt2


def generate_subframe(snr_db=20.0, rng=None):
    """Generate one LTE-5 MHz OFDM subframe at the requested SNR.

    Each active subcarrier carries an independent QPSK symbol per OFDM symbol.
    Complex AWGN is added so that the post-FFT per-active-subcarrier SNR equals
    `snr_db` (linear = 10**(snr_db/10)).

    SNR derivation (for numpy's fft conventions):
        x[n] = ifft(X) scales by 1/N; fft(x) = X exactly.
        If |X[k]|^2 = 1 on an active carrier and the added time-domain noise
        has variance sigma^2 (complex), then after FFT the noise per bin has
        variance N * sigma^2. So per-subcarrier SNR = 1 / (N * sigma^2).
        Therefore sigma^2 = 1 / (N_FFT * snr_linear).

    Args:
        snr_db: target post-FFT per-active-subcarrier SNR in dB.
        rng: numpy Generator (optional). If None, a fresh default_rng() is used.

    Returns:
        rx:         complex array, shape (SAMPLES_PER_SUBFRAME,) — noisy time-domain.
        clean:      complex array, same shape — noise-free time-domain.
        tx_freq:    complex array, shape (N_SYMBOLS_PER_SUBFRAME, N_FFT) —
                    transmitted freq-domain symbols, zeros on guard bins.
        tx_bits:    int8 array, shape (N_SYMBOLS_PER_SUBFRAME, N_ACTIVE, 2) —
                    random bits used, for BER computation downstream.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Random bits per (symbol, subcarrier, 2 bits)
    tx_bits = rng.integers(
        0, 2, size=(N_SYMBOLS_PER_SUBFRAME, N_ACTIVE, 2), dtype=np.int8
    )

    # 2. QPSK map
    tx_data_symbols = qpsk_symbols(tx_bits)          # (14, 300) complex

    # 3. Place into the full N_FFT grid, zeros on DC + guard
    tx_freq = np.zeros((N_SYMBOLS_PER_SUBFRAME, N_FFT), dtype=np.complex128)
    active = np.asarray(active_subcarrier_indices())
    tx_freq[:, active] = tx_data_symbols

    # 4. IFFT each symbol
    tx_time_symbols = np.fft.ifft(tx_freq, axis=1)   # (14, 512)

    # 5. Prepend CP of correct length per symbol, concatenate
    clean_chunks = []
    for sym_idx in range(N_SYMBOLS_PER_SUBFRAME):
        cp_len = cp_length(sym_idx)
        sym = tx_time_symbols[sym_idx]
        clean_chunks.append(np.concatenate([sym[-cp_len:], sym]))
    clean = np.concatenate(clean_chunks)
    assert clean.shape == (SAMPLES_PER_SUBFRAME,), (
        f"expected {SAMPLES_PER_SUBFRAME} samples, got {clean.shape[0]}"
    )

    # 6. AWGN scaled for target per-subcarrier post-FFT SNR
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (N_FFT * snr_lin)
    noise = (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    ) * np.sqrt(noise_var / 2.0)
    rx = clean + noise

    return rx, clean, tx_freq, tx_bits


def generate_subframe_pilots(snr_db=20.0, channel="awgn", rng=None):
    """Generate one subframe with CRS-style pilots and an optional multipath channel.

    Pilots occupy a sparse 2-D grid (see `pilots.py`); data fills the rest.
    The channel (if any) is applied per subcarrier in the frequency domain
    (equivalent to time-domain convolution because the CP absorbs all delays).
    AWGN is added in the time domain with sigma^2 = 1/(N_FFT * snr_lin), so
    `snr_db` means post-FFT per-subcarrier SNR, averaged over the channel
    (E[|H|^2] = 1 by construction).

    Returns:
        rx:        complex array (SAMPLES_PER_SUBFRAME,), noisy time-domain.
        clean:     complex array, noise-free time-domain.
        tx_freq:   (14, N_FFT) complex, pre-channel (pilots + data, guards 0).
        tx_bits:   int8 array (n_data_cells, 2). Order matches
                   `data_mask_active()` ravelled (row-major over (symbol, SC)).
        H:         (N_FFT,) complex channel freq response, or None for AWGN.
    """
    from pilots import pilot_mask_active, data_mask_active, pilot_grid_active
    from channel import sample_channel

    if rng is None:
        rng = np.random.default_rng()

    pilot_mask = pilot_mask_active()
    data_mask = data_mask_active()
    n_data = int(data_mask.sum())

    tx_bits = rng.integers(0, 2, size=(n_data, 2), dtype=np.int8)
    data_symbols = qpsk_symbols(tx_bits)

    grid_active = pilot_grid_active()
    grid_active[data_mask] = data_symbols

    tx_freq = np.zeros((N_SYMBOLS_PER_SUBFRAME, N_FFT), dtype=np.complex128)
    active = np.asarray(active_subcarrier_indices())
    tx_freq[:, active] = grid_active

    H = sample_channel(profile=channel, rng=rng)
    tx_freq_chan = tx_freq * H[np.newaxis, :] if H is not None else tx_freq

    tx_time_symbols = np.fft.ifft(tx_freq_chan, axis=1)
    clean_chunks = []
    for sym_idx in range(N_SYMBOLS_PER_SUBFRAME):
        cp_len = cp_length(sym_idx)
        sym = tx_time_symbols[sym_idx]
        clean_chunks.append(np.concatenate([sym[-cp_len:], sym]))
    clean = np.concatenate(clean_chunks)

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (N_FFT * snr_lin)
    noise = (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    ) * np.sqrt(noise_var / 2.0)
    rx = clean + noise

    return rx, clean, tx_freq, tx_bits, H


def strip_cp_and_fft(rx_or_clean):
    """Recover per-symbol frequency-domain samples from a time-domain subframe.

    Splits the 7680-sample subframe into 14 OFDM symbols (stripping the correct
    CP length from each), runs a 512-pt FFT, and returns a (14, N_FFT) array.

    Args:
        rx_or_clean: complex array of shape (SAMPLES_PER_SUBFRAME,).

    Returns:
        freq: complex array of shape (N_SYMBOLS_PER_SUBFRAME, N_FFT).
    """
    assert rx_or_clean.shape == (SAMPLES_PER_SUBFRAME,)
    out = np.empty((N_SYMBOLS_PER_SUBFRAME, N_FFT), dtype=np.complex128)
    cursor = 0
    for sym_idx in range(N_SYMBOLS_PER_SUBFRAME):
        cp_len = cp_length(sym_idx)
        cursor += cp_len                                  # drop CP
        out[sym_idx] = np.fft.fft(rx_or_clean[cursor:cursor + N_FFT])
        cursor += N_FFT
    assert cursor == SAMPLES_PER_SUBFRAME
    return out
