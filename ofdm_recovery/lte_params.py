"""LTE-5 MHz FDD downlink parameters (normal cyclic prefix).

Single source of truth for dimensions used across the ofdm_recovery package.
Values are taken from the LTE spec for a 5 MHz channel with the 7.68 MHz
sampling rate convention and normal CP.
"""

FS = 7.68e6                    # sample rate (Hz)
N_FFT = 512                    # per-symbol FFT size
SUBCARRIER_SPACING = 15e3      # Hz
N_ACTIVE = 300                 # active subcarriers (25 RBs x 12 SCs)

# Normal CP: first symbol of each slot is longer
N_CP_LONG = 40
N_CP_SHORT = 36

N_SYMBOLS_PER_SLOT = 7
N_SLOTS_PER_SUBFRAME = 2
N_SYMBOLS_PER_SUBFRAME = N_SLOTS_PER_SUBFRAME * N_SYMBOLS_PER_SLOT  # 14

# 2 * (40 + 6*36) + 14 * 512 = 2 * 256 + 7168 = 7680
SAMPLES_PER_SUBFRAME = 7680


def cp_length(symbol_idx):
    """CP length (in samples) for the given intra-subframe symbol index (0..13)."""
    # Symbol 0 of each slot has the long CP; slots start at indices 0 and 7.
    is_first_of_slot = (symbol_idx % N_SYMBOLS_PER_SLOT) == 0
    return N_CP_LONG if is_first_of_slot else N_CP_SHORT


def active_subcarrier_indices():
    """FFT-layout indices that carry data, in order.

    Convention: after a standard FFT (no fftshift), positive baseband freqs
    live at indices 1..N_FFT/2-1 and negative freqs at N_FFT/2..N_FFT-1.
    For LTE-5 MHz we use 150 positive and 150 negative subcarriers around DC,
    skipping DC itself (index 0) and the guard band (indices 151..361).

    Returns a list of 300 integer indices.
    """
    return list(range(1, 151)) + list(range(362, 512))


def guard_subcarrier_indices():
    """Indices that must remain zero: DC (0) and guard band (151..361)."""
    return [0] + list(range(151, 362))
