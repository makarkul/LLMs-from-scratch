"""Evaluate the trained transformer across a range of noise levels (SNR sweep)."""
import argparse
import numpy as np
import torch

from signal_dataset import create_sinusoid_dataloaders
from ts_transformer import TimeSeriesTransformer, TS_TRANSFORMER_CONFIG
from train import evaluate_model


def sweep(model_path, noise_levels, device, val_size=1000, batch_size=32):
    cfg = TS_TRANSFORMER_CONFIG.copy()
    model = TimeSeriesTransformer(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"{'amp_std':>8} {'phase_std':>10} {'in_SNR':>8} {'out_SNR':>9} {'gain':>7} {'MSE':>10}")
    print("-" * 60)
    results = []
    for amp_std, phase_std in noise_levels:
        torch.manual_seed(0)
        np.random.seed(0)
        _, val_loader = create_sinusoid_dataloaders(
            context_length=cfg["context_length"],
            train_size=32, val_size=val_size, batch_size=batch_size,
            amp_noise_std=amp_std, phase_noise_std=phase_std,
        )
        mse, in_snr, out_snr = evaluate_model(model, val_loader, device)
        gain = out_snr - in_snr
        print(f"{amp_std:>8.2f} {phase_std:>10.2f} {in_snr:>8.2f} {out_snr:>9.2f} "
              f"{gain:>+7.2f} {mse:>10.5f}")
        results.append((amp_std, phase_std, in_snr, out_snr, gain, mse))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ts_transformer.pth")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Noise sweep: original (0.2, 0.1) through much harder
    noise_levels = [
        (0.1, 0.05),   # cleaner than training
        (0.2, 0.10),   # training noise
        (0.4, 0.20),   # 2x
        (0.6, 0.30),   # 3x
        (0.8, 0.40),   # 4x
        (1.0, 0.50),   # 5x — very noisy
    ]
    sweep(args.model_path, noise_levels, device)
