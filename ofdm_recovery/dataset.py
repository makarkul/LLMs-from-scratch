"""Batched subframe generator for step-4a training.

Each sample is one LTE-5 MHz subframe: EPA-faded, AWGN, with pilots, on
the post-FFT active-subcarrier grid.

Layout we feed the model:
    rx_grid:   (14, 300, 3) real  = [Re(Y), Im(Y), is_pilot]
    data_mask: (14, 300)    bool  = True on data cells
    tx_bits:   (n_data, 2)  int8  = ground-truth bits in data_mask-ravel order
    snr_db:    scalar              = used for logging, not for the model

SNR is sampled uniformly per-example from [SNR_MIN_DB, SNR_MAX_DB] so the
model sees the full operating regime.
"""

from __future__ import annotations

import numpy as np
import torch

from lte_params import active_subcarrier_indices
from ofdm_signal import generate_subframe_pilots, strip_cp_and_fft
from pilots import pilot_mask_active, data_mask_active


# Curriculum range — covers everything in the classical BER sweep.
SNR_MIN_DB = 0.0
SNR_MAX_DB = 25.0

_ACTIVE = np.asarray(active_subcarrier_indices())
_PILOT_MASK = pilot_mask_active()
_DATA_MASK = data_mask_active()


def _subframe_to_grid(rx):
    """rx (7680,) -> post-FFT active grid (14, 300) complex."""
    freq = strip_cp_and_fft(rx)
    return freq[:, _ACTIVE]


def generate_batch(batch_size: int, rng: np.random.Generator,
                   snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                   channel: str = "epa"):
    """Build one training batch.

    Returns a dict of torch tensors (float32 / bool / int8) on CPU:
        rx_grid:    (B, 14, 300, 3)
        data_mask:  (14, 300)                (shared across batch)
        pilot_mask: (14, 300)                (shared across batch)
        tx_bits:    (B, n_data, 2)           int8 in {0,1}
        snr_db:     (B,) float32
        H:          (B, 512) complex64       true channel, for oracle eval
    """
    n_data = int(_DATA_MASK.sum())
    rx_grid = np.empty((batch_size, 14, 300, 3), dtype=np.float32)
    tx_bits = np.empty((batch_size, n_data, 2), dtype=np.int8)
    H_all = np.empty((batch_size, 512), dtype=np.complex64)
    snrs = np.empty(batch_size, dtype=np.float32)

    pilot_channel = _PILOT_MASK.astype(np.float32)   # (14, 300)

    for b in range(batch_size):
        snr_db = float(rng.uniform(snr_range[0], snr_range[1]))
        rx, _, _, bits, H = generate_subframe_pilots(
            snr_db=snr_db, channel=channel, rng=rng
        )
        grid = _subframe_to_grid(rx)                 # (14, 300) complex
        rx_grid[b, :, :, 0] = grid.real
        rx_grid[b, :, :, 1] = grid.imag
        rx_grid[b, :, :, 2] = pilot_channel
        tx_bits[b] = bits
        snrs[b] = snr_db
        if H is None:
            H_all[b].fill(1.0 + 0j)
        else:
            H_all[b] = H.astype(np.complex64)

    return {
        "rx_grid": torch.from_numpy(rx_grid),
        "data_mask": torch.from_numpy(_DATA_MASK),
        "pilot_mask": torch.from_numpy(_PILOT_MASK),
        "tx_bits": torch.from_numpy(tx_bits),
        "snr_db": torch.from_numpy(snrs),
        "H": torch.from_numpy(H_all),
    }
