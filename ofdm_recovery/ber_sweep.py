"""BER vs SNR sweep for the classical baseline receiver.

Runs many subframes per SNR point, counts bit errors, and plots measured BER
against the theoretical Gray-coded QPSK-in-AWGN curve.

Theory:
    Uncoded QPSK in AWGN with Gray coding:
        BER = Q( sqrt(2 * Eb/N0) )
    For our SNR convention (per-active-subcarrier SNR after FFT), with
    QPSK carrying 2 bits/symbol and unit symbol energy:
        Es/N0 = snr_linear          (by construction of generate_subframe)
        Eb/N0 = Es/N0 / 2           (2 bits per symbol)
    => BER = Q( sqrt(snr_linear) )
"""

import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from ofdm_signal import generate_subframe, generate_subframe_pilots
from baseline_receiver import receive, receive_pilots


def q_func(x):
    """Gaussian Q-function via erfc."""
    return 0.5 * np.array([math.erfc(v / math.sqrt(2.0)) for v in np.atleast_1d(x)])


def theory_ber_qpsk(snr_db):
    snr_lin = 10.0 ** (np.asarray(snr_db) / 10.0)
    # Eb/N0 = Es/N0 / 2 for QPSK; BER = Q(sqrt(2 Eb/N0)) = Q(sqrt(snr_lin))
    return q_func(np.sqrt(snr_lin))


def sweep_epa(snr_points_db, n_subframes_per_point, seed=0):
    """EPA block-fading sweep: measure BER with (a) perfect CSI and (b) LS+interp.

    Returns:
        ber_oracle: (n_snr,) BER with perfect channel knowledge
        ber_lsint:  (n_snr,) BER with LS + linear-freq + nearest-time interp
        n_bits:     (n_snr,) total data bits per SNR point
    """
    rng = np.random.default_rng(seed)
    n = len(snr_points_db)
    ber_oracle = np.empty(n)
    ber_lsint = np.empty(n)
    n_bits_out = np.empty(n, dtype=np.int64)
    for i, snr_db in enumerate(snr_points_db):
        err_o = err_l = bits = 0
        for _ in range(n_subframes_per_point):
            rx, _, _, tx_bits, H = generate_subframe_pilots(
                snr_db=snr_db, channel="epa", rng=rng
            )
            rx_bits_o, _, _ = receive_pilots(rx, H_oracle=H)
            rx_bits_l, _, _ = receive_pilots(rx, H_oracle=None)
            err_o += int(np.sum(rx_bits_o != tx_bits))
            err_l += int(np.sum(rx_bits_l != tx_bits))
            bits += tx_bits.size
        ber_oracle[i] = err_o / max(bits, 1)
        ber_lsint[i] = err_l / max(bits, 1)
        n_bits_out[i] = bits
        print(f"EPA  SNR {snr_db:+5.1f} dB   oracle BER={ber_oracle[i]:.3e}   "
              f"LS+interp BER={ber_lsint[i]:.3e}   bits={bits}")
    return ber_oracle, ber_lsint, n_bits_out


def sweep(snr_points_db, n_subframes_per_point, seed=0):
    rng = np.random.default_rng(seed)
    measured = np.empty(len(snr_points_db))
    n_bits_total = np.empty(len(snr_points_db), dtype=np.int64)
    for i, snr_db in enumerate(snr_points_db):
        errors = 0
        bits = 0
        for _ in range(n_subframes_per_point):
            rx, _, _, tx_bits = generate_subframe(snr_db=snr_db, rng=rng)
            rx_bits, _ = receive(rx)
            errors += int(np.sum(rx_bits != tx_bits))
            bits += tx_bits.size
        measured[i] = errors / max(bits, 1)
        n_bits_total[i] = bits
        print(f"SNR {snr_db:+5.1f} dB   errors={errors:>7d}   bits={bits}   BER={measured[i]:.3e}")
    return measured, n_bits_total


def plot_ber_all(snr_db, awgn_meas, awgn_nbits,
                 epa_oracle, epa_lsint, epa_nbits, save_path):
    """Overlay AWGN (measured + theory) and EPA (oracle + LS-interp) curves."""
    theory = theory_ber_qpsk(snr_db)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.semilogy(snr_db, theory, "k-", linewidth=1.5, label="AWGN theory: Q(√SNR)")

    def _floor(y, n):
        f = 0.5 / np.maximum(n, 1)
        return np.where(y > 0, y, f)

    ax.semilogy(snr_db, _floor(awgn_meas, awgn_nbits), "o",
                color="#065A82", markersize=6, label="AWGN — baseline RX")
    ax.semilogy(snr_db, _floor(epa_oracle, epa_nbits), "s",
                color="#1C7293", markersize=6, label="EPA — perfect CSI")
    ax.semilogy(snr_db, _floor(epa_lsint, epa_nbits), "^",
                color="#B85042", markersize=6, label="EPA — LS + interp")
    ax.set_xlabel("per-subcarrier SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Uncoded QPSK-OFDM — AWGN vs EPA (block-fading, Rayleigh)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_ylim(1e-5, 1.0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ber(snr_db, measured, n_bits, save_path):
    theory = theory_ber_qpsk(snr_db)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(snr_db, theory, "k-", linewidth=1.5, label="theory: Q(√SNR)")
    # Plot measured; replace zeros with a floor so log-scale renders.
    floor = 0.5 / np.maximum(n_bits, 1)
    meas_plot = np.where(measured > 0, measured, floor)
    ax.semilogy(snr_db, meas_plot, "o", color="#065A82", markersize=7,
                label="measured (baseline RX)")
    ax.set_xlabel("per-subcarrier SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Uncoded QPSK-OFDM over AWGN — baseline receiver vs theory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_ylim(1e-6, 1.0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-min", type=float, default=-2.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    parser.add_argument("--n-subframes", type=int, default=50,
                        help="subframes per SNR point")
    parser.add_argument("--out", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channel", type=str, default="both",
                        choices=["awgn", "epa", "both"])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    snr_db = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    awgn_meas = awgn_nbits = None
    epa_oracle = epa_lsint = epa_nbits = None

    if args.channel in ("awgn", "both"):
        print("=== AWGN baseline ===")
        awgn_meas, awgn_nbits = sweep(snr_db, args.n_subframes, seed=args.seed)
    if args.channel in ("epa", "both"):
        print("=== EPA block-fading ===")
        epa_oracle, epa_lsint, epa_nbits = sweep_epa(
            snr_db, args.n_subframes, seed=args.seed + 1
        )

    if args.channel == "awgn":
        plot_ber(snr_db, awgn_meas, awgn_nbits,
                 os.path.join(args.out, "ber_vs_snr.pdf"))
    elif args.channel == "epa":
        # Still produce a comparison fig with just the two EPA curves.
        awgn_meas = np.full_like(snr_db, np.nan)
        awgn_nbits = epa_nbits
        plot_ber_all(snr_db, awgn_meas, awgn_nbits,
                     epa_oracle, epa_lsint, epa_nbits,
                     os.path.join(args.out, "ber_vs_snr_epa.pdf"))
    else:
        plot_ber_all(snr_db, awgn_meas, awgn_nbits,
                     epa_oracle, epa_lsint, epa_nbits,
                     os.path.join(args.out, "ber_vs_snr_all.pdf"))
