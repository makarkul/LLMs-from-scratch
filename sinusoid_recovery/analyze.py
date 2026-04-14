"""Post-training analysis: attention weights, FFT comparison, autoregressive generation."""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from signal_dataset import generate_noisy_sinusoid
from ts_transformer import TimeSeriesTransformer, TS_TRANSFORMER_CONFIG


def get_attention_weights(model, x):
    """Run a forward pass and capture attention weights from all layers/heads.

    Returns a list of (n_layers) tensors, each of shape (batch, n_heads, seq_len, seq_len).
    """
    model.eval()
    attn_weights_all = []

    # Hook to capture attention weights
    hooks = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            # MultiHeadAttention computes attn_weights internally.
            # We need to re-run the attention computation to extract weights.
            pass
        return hook_fn

    # Instead of hooks, manually compute attention weights by running through the model
    with torch.no_grad():
        batch_size, seq_len = x.shape
        x_input = x.unsqueeze(-1)
        tok_embeds = model.input_proj(x_input)
        pos_embeds = model.pos_emb(torch.arange(seq_len, device=x.device))
        h = tok_embeds + pos_embeds

        for block in model.trf_blocks:
            # Pre-norm
            h_norm = block.norm1(h)
            att = block.att

            # Manually compute attention weights
            b, n_tok, d_in = h_norm.shape
            keys = att.W_key(h_norm).view(b, n_tok, att.num_heads, att.head_dim).transpose(1, 2)
            queries = att.W_query(h_norm).view(b, n_tok, att.num_heads, att.head_dim).transpose(1, 2)
            values = att.W_value(h_norm).view(b, n_tok, att.num_heads, att.head_dim).transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = att.mask.bool()[:n_tok, :n_tok]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights_all.append(weights.cpu().numpy())

            # Continue forward pass through the block
            context_vec = (weights @ values).transpose(1, 2).contiguous().view(b, n_tok, att.d_out)
            context_vec = att.out_proj(context_vec)
            h = h + context_vec
            shortcut = h
            h = block.norm2(h)
            h = block.ff(h)
            h = h + shortcut

    return attn_weights_all


def plot_attention_weights(model, device, test_freqs, fs=100.0, save_path="attention_analysis.pdf"):
    """Visualize attention patterns for sinusoids at different frequencies."""
    cfg = model.pos_emb.weight.shape[0]  # context_length
    n_freqs = len(test_freqs)
    n_layers = len(list(model.trf_blocks))

    fig, axes = plt.subplots(n_freqs, n_layers, figsize=(6 * n_layers, 4 * n_freqs))
    if n_freqs == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for i, freq in enumerate(test_freqs):
        # Generate a clean sinusoid (no noise) to see pure attention structure
        clean_signal = np.sin(2 * math.pi * freq * np.arange(cfg) / fs).astype(np.float32)
        x = torch.tensor(clean_signal).unsqueeze(0).to(device)

        attn_weights = get_attention_weights(model, x)

        for layer_idx, weights in enumerate(attn_weights):
            # Average over heads for visualization; shape: (1, n_heads, seq, seq)
            avg_weights = weights[0].mean(axis=0)  # (seq, seq)

            ax = axes[i, layer_idx]
            # Show only the last 32 positions for readability
            display_size = min(32, avg_weights.shape[0])
            im = ax.imshow(
                avg_weights[-display_size:, -display_size:],
                aspect="auto", cmap="viridis"
            )
            ax.set_title(f"Freq={freq:.1f}Hz, Layer {layer_idx+1}")
            ax.set_xlabel("Key position (relative)")
            ax.set_ylabel("Query position (relative)")
            plt.colorbar(im, ax=ax)

    fig.suptitle("Attention Weights by Frequency and Layer", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Attention analysis saved to {save_path}")


def plot_attention_lag_profile(model, device, test_freqs, fs=100.0,
                               save_path="attention_lag_profile.pdf"):
    """For each test frequency, plot attention weight vs lag distance.

    For AR(2), we expect strong weights at lag 1 and lag 2.
    """
    cfg_ctx = model.pos_emb.weight.shape[0]

    fig, axes = plt.subplots(1, len(test_freqs), figsize=(5 * len(test_freqs), 4))
    if len(test_freqs) == 1:
        axes = [axes]

    for i, freq in enumerate(test_freqs):
        clean_signal = np.sin(2 * math.pi * freq * np.arange(cfg_ctx) / fs).astype(np.float32)
        x = torch.tensor(clean_signal).unsqueeze(0).to(device)

        attn_weights = get_attention_weights(model, x)

        # Use the last layer's attention, average over heads
        last_layer_weights = attn_weights[-1][0].mean(axis=0)  # (seq, seq)

        # Compute average attention weight as a function of lag
        max_lag = min(30, cfg_ctx)
        lag_weights = np.zeros(max_lag)
        counts = np.zeros(max_lag)

        for q in range(cfg_ctx):
            for k in range(q + 1):  # causal: can only attend to k <= q
                lag = q - k
                if lag < max_lag:
                    lag_weights[lag] += last_layer_weights[q, k]
                    counts[lag] += 1

        lag_weights /= np.maximum(counts, 1)

        # Theoretical AR(2) coefficient
        omega = 2 * math.pi * freq / fs
        ar1_coeff = 2 * math.cos(omega)

        ax = axes[i]
        ax.bar(range(max_lag), lag_weights, alpha=0.7, label="Learned attention")
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="Lag 1")
        ax.axvline(x=2, color="orange", linestyle="--", alpha=0.5, label="Lag 2")
        ax.set_xlabel("Lag (samples)")
        ax.set_ylabel("Avg Attention Weight")
        ax.set_title(f"f={freq:.1f}Hz (AR coeff: 2cos(w)={ar1_coeff:.3f})")
        ax.legend(fontsize=8)

    fig.suptitle("Attention Weight vs Lag (expect peaks at lag 1 & 2 for AR(2))", fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Lag profile saved to {save_path}")


def plot_denoising_comparison(model, device, test_freq=5.0, fs=100.0,
                               amp_noise_std=0.2, phase_noise_std=0.1,
                               save_path="denoising_comparison.pdf"):
    """Plot noisy input, model prediction, and clean ground truth side by side."""
    cfg_ctx = model.pos_emb.weight.shape[0]
    model.eval()

    noisy, clean = generate_noisy_sinusoid(
        cfg_ctx + 1, test_freq, 1.0, 0.0, fs, amp_noise_std, phase_noise_std
    )

    x = torch.tensor(noisy[:cfg_ctx]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()

    clean_target = clean[1:cfg_ctx + 1]
    t = np.arange(cfg_ctx) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain
    ax1.plot(t, noisy[:cfg_ctx], alpha=0.5, label="Noisy input", linewidth=0.8)
    ax1.plot(t, pred, label="Model prediction", linewidth=1.5)
    ax1.plot(t, clean_target, "--", label="Clean target", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Denoising: f={test_freq:.1f}Hz")
    ax1.legend()

    # Frequency domain
    n_fft = cfg_ctx
    freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    noisy_fft = np.abs(np.fft.rfft(noisy[:cfg_ctx])) / n_fft
    pred_fft = np.abs(np.fft.rfft(pred)) / n_fft
    clean_fft = np.abs(np.fft.rfft(clean_target)) / n_fft

    ax2.plot(freqs, noisy_fft, alpha=0.5, label="Noisy input")
    ax2.plot(freqs, pred_fft, label="Model prediction")
    ax2.plot(freqs, clean_fft, "--", label="Clean target", alpha=0.8)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|FFT|")
    ax2.set_title("Frequency Spectrum Comparison")
    ax2.legend()
    ax2.set_xlim(0, 30)

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Denoising comparison saved to {save_path}")


def autoregressive_generation(model, device, seed_signal, n_generate=200):
    """Feed model its own predictions to generate an extended signal.

    Args:
        model: trained TimeSeriesTransformer
        seed_signal: 1D numpy array of initial samples (length >= context_length)
        n_generate: number of new samples to generate

    Returns:
        generated: numpy array of seed + generated samples
    """
    model.eval()
    cfg_ctx = model.pos_emb.weight.shape[0]
    generated = list(seed_signal)

    with torch.no_grad():
        for _ in range(n_generate):
            # Use last context_length samples
            window = torch.tensor(
                generated[-cfg_ctx:], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = model(window)
            next_sample = pred[0, -1].item()  # prediction at last position
            generated.append(next_sample)

    return np.array(generated)


def plot_autoregressive_generation(model, device, test_freq=5.0, fs=100.0,
                                    n_generate=300,
                                    save_path="autoregressive_generation.pdf"):
    """Generate a sinusoid autoregressively from a noisy seed."""
    cfg_ctx = model.pos_emb.weight.shape[0]

    # Create a noisy seed
    noisy_seed, _ = generate_noisy_sinusoid(
        cfg_ctx, test_freq, 1.0, 0.0, fs, amp_noise_std=0.2, phase_noise_std=0.1
    )

    generated = autoregressive_generation(model, device, noisy_seed, n_generate)

    # Ground truth extended signal
    total_len = cfg_ctx + n_generate
    t = np.arange(total_len) / fs
    clean_extended = np.sin(2 * math.pi * test_freq * t)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t[:cfg_ctx], noisy_seed, alpha=0.4, label="Noisy seed", color="gray")
    ax.plot(t, generated, label="Autoregressive output", linewidth=1.2)
    ax.plot(t, clean_extended, "--", label="Clean reference", alpha=0.7)
    ax.axvline(x=cfg_ctx / fs, color="red", linestyle=":", label="Seed/generation boundary")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Autoregressive Generation from Noisy Seed (f={test_freq:.1f}Hz)")
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Autoregressive generation plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained time-series transformer")
    parser.add_argument("--model-path", type=str, default="ts_transformer.pth")
    parser.add_argument("--test-freqs", type=float, nargs="+", default=[2.0, 5.0, 10.0, 18.0])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TS_TRANSFORMER_CONFIG.copy()
    model = TimeSeriesTransformer(cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    print("\n1. Attention weight analysis...")
    plot_attention_weights(model, device, args.test_freqs)

    print("\n2. Attention lag profile...")
    plot_attention_lag_profile(model, device, args.test_freqs)

    print("\n3. Denoising comparison...")
    for freq in args.test_freqs:
        plot_denoising_comparison(
            model, device, test_freq=freq,
            save_path=f"denoising_{freq:.0f}Hz.pdf"
        )

    print("\n4. Autoregressive generation...")
    for freq in [3.0, 7.0, 15.0]:
        plot_autoregressive_generation(
            model, device, test_freq=freq,
            save_path=f"autoreg_{freq:.0f}Hz.pdf"
        )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
