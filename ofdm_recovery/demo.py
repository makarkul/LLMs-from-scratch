"""End-to-end demo: generate one LTE-5 MHz subframe, plot, and sanity-check.

Usage:
    python demo.py --snr 20 --out figures/
"""

import argparse
import os
import numpy as np

from lte_params import (
    SAMPLES_PER_SUBFRAME,
    N_ACTIVE,
    N_FFT,
    active_subcarrier_indices,
    guard_subcarrier_indices,
)
from ofdm_signal import generate_subframe, strip_cp_and_fft
from visualize import plot_time_domain, plot_spectrogram, plot_constellation


def run(snr_db, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    rx, clean, tx_freq, tx_bits = generate_subframe(snr_db=snr_db, rng=rng)

    # --- Sanity checks (printed, not asserted so demo always runs) ---
    print("=" * 60)
    print(f"LTE-5 MHz subframe @ {snr_db:.1f} dB SNR  (seed={seed})")
    print("=" * 60)
    print(f"sample count          : {rx.shape[0]}  "
          f"(expected {SAMPLES_PER_SUBFRAME})")

    active = np.asarray(active_subcarrier_indices())
    guards = np.asarray(guard_subcarrier_indices())

    # 1. Guard bands are exactly zero in the tx freq grid
    guard_energy_tx = np.max(np.abs(tx_freq[:, guards]))
    print(f"tx guard max |X|      : {guard_energy_tx:.3e}  (expect 0)")

    # 2. Clean round-trip: FFT of clean time-domain should match tx_freq on
    #    active subcarriers (identity with numpy's fft/ifft conventions).
    clean_freq = strip_cp_and_fft(clean)
    roundtrip_err = np.max(np.abs(clean_freq[:, active] - tx_freq[:, active]))
    print(f"round-trip max err    : {roundtrip_err:.3e}  (expect ~1e-12)")

    # 3. Measured per-active-subcarrier SNR
    rx_freq = strip_cp_and_fft(rx)
    noise_on_active = rx_freq[:, active] - tx_freq[:, active]
    meas_snr_lin = (
        np.mean(np.abs(tx_freq[:, active]) ** 2)
        / np.mean(np.abs(noise_on_active) ** 2)
    )
    meas_snr_db = 10 * np.log10(meas_snr_lin)
    print(f"measured per-SC SNR   : {meas_snr_db:+.2f} dB  "
          f"(target {snr_db:+.1f} dB)")

    # 4. Noise-floor on guard bins after FFT
    noise_on_guards = rx_freq[:, guards]
    guard_noise_power = np.mean(np.abs(noise_on_guards) ** 2)
    print(f"rx guard noise power  : {guard_noise_power:.3e}")

    # 5. Parseval sanity: sum |X|^2 over active == N_FFT^2 * mean|x|^2 / N_FFT
    #    (numpy conv: mean|x|^2 = (1/N^2) * sum|X|^2 after ifft)
    sum_X_sq = np.sum(np.abs(tx_freq) ** 2)
    # Mean over one useful symbol (skip CP to keep the identity clean)
    sym0 = np.fft.ifft(tx_freq[0])
    lhs = N_FFT ** 2 * np.mean(np.abs(sym0) ** 2)
    rhs = np.sum(np.abs(tx_freq[0]) ** 2)
    print(f"Parseval (symbol 0)   : LHS {lhs:.3f}  RHS {rhs:.3f}  "
          f"(should be equal)")
    print(f"total tx freq energy  : {sum_X_sq:.1f}  "
          f"(expected {14 * N_ACTIVE} = 14x300)")
    print("=" * 60)

    # --- Figures ---
    plot_time_domain(rx, clean, os.path.join(out_dir, "time_domain.pdf"))
    plot_spectrogram(rx, os.path.join(out_dir, "spectrogram.pdf"))
    plot_constellation(rx, os.path.join(out_dir, "constellation.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr", type=float, default=20.0,
                        help="post-FFT per-subcarrier SNR in dB (default 20)")
    parser.add_argument("--out", type=str, default="figures",
                        help="output directory for PDFs")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()
    run(args.snr, args.out, args.seed)
