"""BER sweep for the trained learned receiver, overlaid on classical curves.

Loads a checkpoint, runs N subframes per SNR point through the model, and
plots the result against AWGN theory + EPA perfect-CSI + EPA LS+interp.
"""

import argparse
import os
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import generate_batch
from model import LearnedReceiver
from ber_sweep import sweep, sweep_epa, theory_ber_qpsk


def learned_ber(model, device, snr_db, n_subframes, batch_size, seed):
    """Evaluate the model BER at a fixed SNR over n_subframes."""
    rng = np.random.default_rng(seed)
    errs = bits = 0
    remaining = n_subframes
    model.eval()
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            batch = generate_batch(b, rng, snr_range=(snr_db, snr_db))
            rx = batch["rx_grid"].to(device)
            tx = batch["tx_bits"].to(device)
            mask = batch["data_mask"].to(device)
            logits = model(rx)                          # (B, 14, 300, 2)
            pred = (logits > 0).to(torch.int8)
            # Pull out data cells
            idx = torch.nonzero(mask.view(-1), as_tuple=False).squeeze(-1)
            pred_flat = pred.view(b, -1, 2)[:, idx, :]
            errs += int((pred_flat != tx).sum().item())
            bits += tx.numel()
            remaining -= b
    return errs / max(bits, 1), bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=25.0)
    p.add_argument("--snr-step", type=float, default=2.5)
    p.add_argument("--n-subframes", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="figures")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-classical", action="store_true",
                   help="Skip re-running classical AWGN/EPA baselines (saves time)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cargs = ckpt["args"]
    model = LearnedReceiver(
        d_model=cargs["d_model"], n_heads=cargs["n_heads"],
        n_layers=cargs["n_layers"], d_ff=cargs["d_ff"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded ckpt step {ckpt['step']}  best val BER {ckpt['val_ber']:.3e}")

    snr_db = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    print(f"SNR points: {snr_db}")

    # --- learned receiver ---
    print("=== Learned receiver BER ===")
    learned_ber_arr = np.empty(len(snr_db))
    learned_nbits = np.empty(len(snr_db), dtype=np.int64)
    for i, s in enumerate(snr_db):
        ber, nb = learned_ber(model, device, float(s),
                              args.n_subframes, args.batch_size,
                              seed=args.seed + i)
        learned_ber_arr[i] = ber
        learned_nbits[i] = nb
        print(f"SNR {s:+5.1f} dB   learned BER {ber:.3e}   bits {nb}")

    # --- classical baselines ---
    if args.skip_classical:
        awgn_meas = np.full_like(snr_db, np.nan)
        awgn_nbits = np.ones_like(snr_db, dtype=np.int64)
        epa_oracle = np.full_like(snr_db, np.nan)
        epa_lsint = np.full_like(snr_db, np.nan)
        epa_nbits = np.ones_like(snr_db, dtype=np.int64)
    else:
        print("=== AWGN baseline ===")
        awgn_meas, awgn_nbits = sweep(snr_db, args.n_subframes, seed=args.seed + 100)
        print("=== EPA baselines ===")
        epa_oracle, epa_lsint, epa_nbits = sweep_epa(
            snr_db, args.n_subframes, seed=args.seed + 200
        )

    # --- plot ---
    theory = theory_ber_qpsk(snr_db)

    def _floor(y, n):
        f = 0.5 / np.maximum(n, 1)
        return np.where(y > 0, y, f)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.semilogy(snr_db, theory, "k-", linewidth=1.5, label="AWGN theory")
    if not args.skip_classical:
        ax.semilogy(snr_db, _floor(awgn_meas, awgn_nbits), "o",
                    color="#065A82", markersize=6, label="AWGN baseline")
        ax.semilogy(snr_db, _floor(epa_oracle, epa_nbits), "s",
                    color="#1C7293", markersize=6, label="EPA perfect CSI")
        ax.semilogy(snr_db, _floor(epa_lsint, epa_nbits), "^",
                    color="#B85042", markersize=6, label="EPA LS+interp")
    ax.semilogy(snr_db, _floor(learned_ber_arr, learned_nbits), "D",
                color="#2C5F2D", markersize=7,
                label="EPA learned RX (step 4a)")

    ax.set_xlabel("per-subcarrier SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Step 4a: learned OFDM receiver vs classical baselines")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_ylim(1e-5, 1.0)
    fig.tight_layout()
    out_path = os.path.join(args.out, "ber_learned.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Dump numbers for the README
    print("\nSNR(dB)   learned   oracle    LS+int    AWGN")
    for i, s in enumerate(snr_db):
        a = awgn_meas[i] if not args.skip_classical else float("nan")
        o = epa_oracle[i] if not args.skip_classical else float("nan")
        l = epa_lsint[i] if not args.skip_classical else float("nan")
        print(f"{s:+6.1f}   {learned_ber_arr[i]:.3e}   {o:.3e}   {l:.3e}   {a:.3e}")


if __name__ == "__main__":
    main()
