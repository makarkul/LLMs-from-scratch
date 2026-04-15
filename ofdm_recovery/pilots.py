"""Pilot pattern for channel estimation.

Simplified LTE-inspired CRS (cell reference signal) layout:
    * Pilots live on OFDM symbols {0, 4, 7, 11} of the subframe
    * Within a pilot symbol, every 6th active subcarrier carries a pilot
      (positions 0, 6, 12, ..., 294 in the 300-active index space).
    * Pilot value is a fixed QPSK point (1+1j)/sqrt(2), known at the receiver.
    * Pilot subcarriers carry NO data on pilot symbols.
    * Non-pilot symbols (1,2,3,5,6,8,9,10,12,13) carry data on all 300 SCs.

Counts per subframe:
    pilots:     4 symbols x 50 subcarriers = 200
    data cells: 10 * 300 + 4 * 250 = 3000 + 1000 = 4000
"""

import numpy as np

from lte_params import N_ACTIVE, N_SYMBOLS_PER_SUBFRAME

PILOT_SYMBOLS = (0, 4, 7, 11)
PILOT_SC_STRIDE = 6
PILOT_VALUE = (1.0 + 1.0j) / np.sqrt(2.0)


def pilot_active_indices():
    """Indices within the 300-active-subcarrier array that carry pilots."""
    return np.arange(0, N_ACTIVE, PILOT_SC_STRIDE)   # 0, 6, 12, ..., 294  -> 50


def pilot_mask_active():
    """Boolean mask of shape (14, 300). True where a pilot lives."""
    mask = np.zeros((N_SYMBOLS_PER_SUBFRAME, N_ACTIVE), dtype=bool)
    pilot_sc = pilot_active_indices()
    for sym in PILOT_SYMBOLS:
        mask[sym, pilot_sc] = True
    return mask


def data_mask_active():
    """Boolean mask of shape (14, 300). True where user data lives."""
    return ~pilot_mask_active()


def pilot_grid_active():
    """Complex grid of shape (14, 300): pilot value where mask is True, else 0."""
    grid = np.zeros((N_SYMBOLS_PER_SUBFRAME, N_ACTIVE), dtype=np.complex128)
    grid[pilot_mask_active()] = PILOT_VALUE
    return grid
