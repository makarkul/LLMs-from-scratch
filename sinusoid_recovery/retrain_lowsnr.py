"""Retrain at a harder noise level with two configurations."""
import argparse
import time
import torch
import numpy as np

from signal_dataset import create_sinusoid_dataloaders
from ts_transformer import TimeSeriesTransformer, count_parameters
from train import train, evaluate_model


BASE_CFG = {
    "context_length": 128,
    "emb_dim": 32,        # 2 heads * 16
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.05,
    "qkv_bias": False,
}

SCALED_CFG = {
    "context_length": 256,   # longer window -> more noise averaging
    "emb_dim": 72,           # 4 heads * 18  (18 ~ 2*log2(256))
    "n_heads": 4,            # more patterns (AR + smoothing taps)
    "n_layers": 3,           # iterative refinement depth
    "drop_rate": 0.05,
    "qkv_bias": False,
}


def run(cfg, tag, epochs, amp_std, phase_std, train_size, val_size, batch_size, lr, device):
    torch.manual_seed(42)
    np.random.seed(42)
    model = TimeSeriesTransformer(cfg).to(device)
    print(f"\n=== {tag} ===")
    print(f"Config: {cfg}")
    print(f"Parameters: {count_parameters(model):,}")

    train_loader, val_loader = create_sinusoid_dataloaders(
        context_length=cfg["context_length"],
        train_size=train_size, val_size=val_size, batch_size=batch_size,
        amp_noise_std=amp_std, phase_noise_std=phase_std,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    train(model, train_loader, val_loader, optimizer, device,
          num_epochs=epochs, eval_freq=100, eval_iter=8)
    print(f"Training time: {time.time() - t0:.1f}s")
    mse, in_snr, out_snr = evaluate_model(model, val_loader, device)
    print(f"FINAL @ noise ({amp_std},{phase_std}): "
          f"in_SNR={in_snr:.2f} dB, out_SNR={out_snr:.2f} dB, "
          f"gain={out_snr-in_snr:+.2f} dB, MSE={mse:.5f}")
    return model, (in_snr, out_snr, mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-std", type=float, default=0.8)
    parser.add_argument("--phase-std", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run(BASE_CFG, "A) Base arch @ hard noise",
        args.epochs, args.amp_std, args.phase_std,
        args.train_size, args.val_size, args.batch_size, args.lr, device)

    run(SCALED_CFG, "B) Scaled arch @ hard noise",
        args.epochs, args.amp_std, args.phase_std,
        args.train_size, args.val_size, args.batch_size, args.lr, device)
