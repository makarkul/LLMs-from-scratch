import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from signal_dataset import create_sinusoid_dataloaders
from ts_transformer import TimeSeriesTransformer, TS_TRANSFORMER_CONFIG, count_parameters


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    predictions = model(input_batch)
    loss = torch.nn.functional.mse_loss(predictions, target_batch)
    return loss


def calc_snr_improvement(input_batch, target_batch, predictions):
    """Compute SNR improvement: how much better the predictions are than the noisy input.

    SNR = 10 * log10(signal_power / noise_power)
    We compare:
        - input SNR:  clean target vs noisy input (shifted to align)
        - output SNR: clean target vs model predictions
    """
    # Model predicts target[t] = clean(t+1) from input[0..t] = noisy(0..t).
    # Fair baseline: "naive predictor" uses noisy(t) to predict clean(t+1).
    clean = target_batch.detach().cpu().numpy()
    pred = predictions.detach().cpu().numpy()
    noisy_input = input_batch.detach().cpu().numpy()

    signal_power = np.mean(clean ** 2)

    # Naive predictor: use noisy[t] as prediction for clean[t+1] (= target[t])
    naive_error = noisy_input - clean
    input_noise_power = np.mean(naive_error ** 2)
    input_snr = 10 * np.log10(signal_power / (input_noise_power + 1e-10))

    # Model predictor
    model_error = pred - clean
    output_noise_power = np.mean(model_error ** 2)
    output_snr = 10 * np.log10(signal_power / (output_noise_power + 1e-10))

    return input_snr, output_snr


def evaluate_model(model, data_loader, device, num_batches=None):
    model.eval()
    total_loss = 0.0
    total_input_snr = 0.0
    total_output_snr = 0.0
    count = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            predictions = model(input_batch)
            loss = torch.nn.functional.mse_loss(predictions, target_batch)
            total_loss += loss.item()

            in_snr, out_snr = calc_snr_improvement(input_batch, target_batch, predictions)
            total_input_snr += in_snr
            total_output_snr += out_snr
            count += 1

    model.train()
    avg_loss = total_loss / count
    avg_in_snr = total_input_snr / count
    avg_out_snr = total_output_snr / count
    return avg_loss, avg_in_snr, avg_out_snr


def train(model, train_loader, val_loader, optimizer, device, num_epochs,
          eval_freq=50, eval_iter=10):
    train_losses = []
    val_losses = []
    val_snr_improvements = []
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, _, _ = evaluate_model(model, train_loader, device, num_batches=eval_iter)
                val_loss, val_in_snr, val_out_snr = evaluate_model(
                    model, val_loader, device, num_batches=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_snr_improvements.append(val_out_snr - val_in_snr)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train MSE {train_loss:.6f}, Val MSE {val_loss:.6f}, "
                      f"SNR improvement {val_out_snr - val_in_snr:+.1f} dB "
                      f"(input {val_in_snr:.1f} -> output {val_out_snr:.1f} dB)")

    return train_losses, val_losses, val_snr_improvements


def plot_losses(train_losses, val_losses, snr_improvements, save_path="training_results.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps = range(len(train_losses))
    ax1.plot(steps, train_losses, label="Train MSE")
    ax1.plot(steps, val_losses, linestyle="-.", label="Val MSE")
    ax1.set_xlabel("Eval Steps")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(steps, snr_improvements, color="green")
    ax2.set_xlabel("Eval Steps")
    ax2.set_ylabel("SNR Improvement (dB)")
    ax2.set_title("Denoising Performance")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train time-series transformer for sinusoid recovery")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--amp-noise", type=float, default=0.2)
    parser.add_argument("--phase-noise", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--save-model", type=str, default="ts_transformer.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = TS_TRANSFORMER_CONFIG.copy()
    print(f"Model config: {cfg}")

    torch.manual_seed(42)
    model = TimeSeriesTransformer(cfg).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader, val_loader = create_sinusoid_dataloaders(
        context_length=cfg["context_length"],
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        amp_noise_std=args.amp_noise,
        phase_noise_std=args.phase_noise,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 80)

    train_losses, val_losses, snr_improvements = train(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=args.eval_freq
    )

    plot_losses(train_losses, val_losses, snr_improvements)

    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    val_loss, val_in_snr, val_out_snr = evaluate_model(model, val_loader, device)
    print(f"  Val MSE:           {val_loss:.6f}")
    print(f"  Input SNR:         {val_in_snr:.1f} dB")
    print(f"  Output SNR:        {val_out_snr:.1f} dB")
    print(f"  SNR Improvement:   {val_out_snr - val_in_snr:+.1f} dB")


if __name__ == "__main__":
    main()
