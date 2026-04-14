"""Compare scalar-embedding vs FFT-embedding transformers at matched param budget."""
import argparse
import time
import torch
import numpy as np

from signal_dataset import create_sinusoid_dataloaders
from ts_transformer import TimeSeriesTransformer, TS_TRANSFORMER_CONFIG
from ts_transformer_fft import TimeSeriesTransformerFFT, TS_TRANSFORMER_FFT_CONFIG
from train import train, evaluate_model


def run(model, cfg, tag, epochs, amp_std, phase_std,
        train_size, val_size, batch_size, lr, device):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {tag} ({n:,} params) ===")
    train_loader, val_loader = create_sinusoid_dataloaders(
        context_length=cfg["context_length"],
        train_size=train_size, val_size=val_size, batch_size=batch_size,
        amp_noise_std=amp_std, phase_noise_std=phase_std,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    train(model, train_loader, val_loader, opt, device,
          num_epochs=epochs, eval_freq=100, eval_iter=8)
    elapsed = time.time() - t0
    mse, in_snr, out_snr = evaluate_model(model, val_loader, device)
    print(f"FINAL: in_SNR={in_snr:.2f}, out_SNR={out_snr:.2f}, "
          f"gain={out_snr - in_snr:+.2f} dB, MSE={mse:.5f}, time={elapsed:.0f}s")
    return in_snr, out_snr, mse, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-std", type=float, default=0.2)
    parser.add_argument("--phase-std", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  noise=({args.amp_std},{args.phase_std})  epochs={args.epochs}")

    torch.manual_seed(42); np.random.seed(42)
    m1 = TimeSeriesTransformer(TS_TRANSFORMER_CONFIG.copy()).to(device)
    r1 = run(m1, TS_TRANSFORMER_CONFIG, "Scalar embedding",
             args.epochs, args.amp_std, args.phase_std,
             args.train_size, args.val_size, args.batch_size, args.lr, device)

    torch.manual_seed(42); np.random.seed(42)
    m2 = TimeSeriesTransformerFFT(TS_TRANSFORMER_FFT_CONFIG.copy()).to(device)
    r2 = run(m2, TS_TRANSFORMER_FFT_CONFIG, "FFT embedding",
             args.epochs, args.amp_std, args.phase_std,
             args.train_size, args.val_size, args.batch_size, args.lr, device)

    print("\n" + "=" * 60)
    print(f"Scalar: gain {r1[1]-r1[0]:+.2f} dB  MSE {r1[2]:.5f}  {r1[3]:.0f}s")
    print(f"FFT   : gain {r2[1]-r2[0]:+.2f} dB  MSE {r2[2]:.5f}  {r2[3]:.0f}s")
