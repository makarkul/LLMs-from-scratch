"""Classical OFDM receiver for the AWGN case.

No channel estimation (AWGN has nothing to estimate). No soft bits. Just:
    strip CP  ->  FFT  ->  pick active subcarriers  ->  QPSK hard-slice  ->  bits

This is the reference any learned receiver has to match on this task.
"""

import numpy as np

from lte_params import active_subcarrier_indices
from ofdm_signal import strip_cp_and_fft


def qpsk_slice(symbols):
    """Hard-decision QPSK demap.

    Inverse of the Gray map in ofdm_signal.qpsk_symbols:
        re > 0 -> bit0 = 0,  re < 0 -> bit0 = 1
        im > 0 -> bit1 = 0,  im < 0 -> bit1 = 1

    Args:
        symbols: complex array of any shape.

    Returns:
        int8 array with a trailing size-2 dim (bit0, bit1) per symbol.
    """
    bits = np.empty(symbols.shape + (2,), dtype=np.int8)
    bits[..., 0] = (symbols.real < 0).astype(np.int8)
    bits[..., 1] = (symbols.imag < 0).astype(np.int8)
    return bits


def ls_channel_estimate(rx_freq_active):
    """LS channel estimate on the active-subcarrier grid, then interpolate.

    Freq direction: linear interpolation across the 300 active subcarriers
        from the 50 pilot positions (per pilot symbol).
    Time direction: nearest pilot symbol (symbols 0, 4, 7, 11 define the
        four estimates; every other OFDM symbol copies the nearest one).

    Args:
        rx_freq_active: complex (14, 300) freq-domain grid on active SCs.

    Returns:
        H_est_active: complex (14, 300) channel estimate on active SCs.
    """
    from pilots import PILOT_SYMBOLS, PILOT_VALUE, pilot_active_indices
    from lte_params import N_ACTIVE, N_SYMBOLS_PER_SUBFRAME

    pilot_sc = pilot_active_indices()                 # shape (50,)
    # LS at pilot cells: H_hat = Y / X_pilot
    H_pilot = rx_freq_active[np.ix_(PILOT_SYMBOLS, pilot_sc)] / PILOT_VALUE
    # shape (4, 50)

    # Frequency interpolation: for each pilot symbol, linearly interp over
    # all 300 active subcarriers.
    all_sc = np.arange(N_ACTIVE)
    H_pilot_symbols = np.empty((len(PILOT_SYMBOLS), N_ACTIVE), dtype=np.complex128)
    for i in range(len(PILOT_SYMBOLS)):
        H_pilot_symbols[i].real = np.interp(all_sc, pilot_sc, H_pilot[i].real)
        H_pilot_symbols[i].imag = np.interp(all_sc, pilot_sc, H_pilot[i].imag)

    # Time interpolation: nearest pilot symbol for each of the 14 symbols.
    pilot_syms = np.asarray(PILOT_SYMBOLS)
    H_est = np.empty((N_SYMBOLS_PER_SUBFRAME, N_ACTIVE), dtype=np.complex128)
    for sym in range(N_SYMBOLS_PER_SUBFRAME):
        nearest = np.argmin(np.abs(pilot_syms - sym))
        H_est[sym] = H_pilot_symbols[nearest]
    return H_est


def receive_pilots(rx, H_oracle=None):
    """Pilot-aware receiver with optional perfect CSI for an oracle baseline.

    Args:
        rx: complex (SAMPLES_PER_SUBFRAME,).
        H_oracle: if given, (N_FFT,) true channel freq response — skip
                  estimation and use perfect CSI. For AWGN pass all-ones or
                  leave as None (means treat the channel as 1).

    Returns:
        rx_bits_data: int8 (n_data_cells, 2) bits in the same order as
                      `generate_subframe_pilots` returns tx_bits.
        rx_symbols_eq: complex (14, 300) — equalized active grid.
        H_est_active: complex (14, 300) — channel estimate used.
    """
    from pilots import data_mask_active
    from lte_params import active_subcarrier_indices

    active = np.asarray(active_subcarrier_indices())
    rx_freq = strip_cp_and_fft(rx)
    rx_freq_active = rx_freq[:, active]

    if H_oracle is not None:
        H_active = H_oracle[active]
        H_est_active = np.broadcast_to(H_active, rx_freq_active.shape).copy()
    else:
        H_est_active = ls_channel_estimate(rx_freq_active)

    # Zero-forcing equalize
    rx_symbols_eq = rx_freq_active / H_est_active

    data_mask = data_mask_active()
    data_symbols = rx_symbols_eq[data_mask]
    rx_bits_data = qpsk_slice(data_symbols)
    return rx_bits_data, rx_symbols_eq, H_est_active


def receive(rx):
    """Full baseline receive chain: rx time-domain -> bits.

    Args:
        rx: complex array of shape (SAMPLES_PER_SUBFRAME,).

    Returns:
        rx_bits:   int8 array of shape (N_SYMBOLS_PER_SUBFRAME, N_ACTIVE, 2).
        rx_symbols: complex array of shape (N_SYMBOLS_PER_SUBFRAME, N_ACTIVE).
    """
    active = np.asarray(active_subcarrier_indices())
    rx_freq = strip_cp_and_fft(rx)
    rx_symbols = rx_freq[:, active]
    rx_bits = qpsk_slice(rx_symbols)
    return rx_bits, rx_symbols
