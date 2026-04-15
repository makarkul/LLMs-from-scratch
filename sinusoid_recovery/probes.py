"""Linear probes on the trained TimeSeriesTransformer.

Question: did the next-sample-prediction transformer implicitly learn
frequency / period even though it was never asked about it?

Method: freeze the trained model. Feed many noisy sinusoids with known
frequency. At each transformer layer, grab the hidden state. Train a
*separate* Linear(emb_dim, 1) head to predict frequency from that hidden
state. If the linear probe is accurate, frequency was linearly readable
from the model's internal representation.

Outputs:
    - Printed R^2 and RMSE of each probe
    - probes_summary.pdf: scatter of true vs predicted frequency per layer

Run:
    python probes.py --ckpt ts_transformer.pth
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from signal_dataset import generate_noisy_sinusoid
from ts_transformer import TimeSeriesTransformer, TS_TRANSFORMER_CONFIG


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class HiddenStateCollector:
    """Registers forward hooks to capture outputs of each transformer block
    plus the input embedding, so we can probe multiple depths."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

        # Block 0 output, block 1 output (our config has 2 blocks)
        for i, blk in enumerate(model.trf_blocks):
            h = blk.register_forward_hook(self._make_hook(f"layer_{i+1}"))
            self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.activations[name] = out.detach()
        return hook

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def generate_feature_bank(model, n_samples, freq_range, fs, context_length,
                          amp_range=(0.5, 2.0),
                          amp_noise_std=0.2, phase_noise_std=0.1, seed=0):
    """Produce (features_per_layer, frequencies) for probe training/eval.

    For each sample: draw random f, A, phi; make a noisy sinusoid; run
    through the frozen model; capture hidden states at each layer; average
    across positions (simple invariant: probe sees a fixed-length feature).
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    collector = HiddenStateCollector(model)
    model.eval()

    freqs = np.empty(n_samples, dtype=np.float32)
    features = {}                               # name -> (n_samples, emb_dim)

    with torch.no_grad():
        for i in range(n_samples):
            f = float(rng.uniform(*freq_range))
            a = float(rng.uniform(*amp_range))
            phi = float(rng.uniform(0.0, 2 * math.pi))
            noisy, _ = generate_noisy_sinusoid(
                context_length, f, a, phi, fs,
                amp_noise_std=amp_noise_std,
                phase_noise_std=phase_noise_std,
            )
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            _ = model(x)

            for name, act in collector.activations.items():
                # act: (1, seq_len, emb_dim) -> mean over seq_len -> (emb_dim,)
                vec = act.mean(dim=1).squeeze(0).cpu().numpy()
                features.setdefault(name, np.empty((n_samples, vec.shape[-1]),
                                                   dtype=np.float32))
                features[name][i] = vec
            freqs[i] = f
            collector.clear()

    collector.remove()
    return features, freqs


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def train_linear_probe(X_train, y_train, X_test, y_test,
                       steps=2000, lr=1e-2, weight_decay=1e-4, device="cpu"):
    """Simple L2 linear regression via closed form (ridge) + report test fit.

    Closed-form ridge has no hyperparameter search to muddy the message.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Add bias column
    def augment(X):
        ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        return torch.cat([X, ones], dim=1)

    Xa = augment(X_train_t)
    Xt = augment(X_test_t)

    # Ridge: w = (X^T X + lam I)^(-1) X^T y
    lam = weight_decay * Xa.shape[0]
    eye = torch.eye(Xa.shape[1], device=device)
    w = torch.linalg.solve(Xa.T @ Xa + lam * eye, Xa.T @ y_train_t)

    pred_train = (Xa @ w).cpu().numpy()
    pred_test = (Xt @ w).cpu().numpy()

    def r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / max(ss_tot, 1e-12)

    def rmse(y, yhat):
        return float(np.sqrt(np.mean((y - yhat) ** 2)))

    return {
        "r2_train": r2(y_train, pred_train),
        "r2_test": r2(y_test, pred_test),
        "rmse_train": rmse(y_train, pred_train),
        "rmse_test": rmse(y_test, pred_test),
        "pred_test": pred_test,
        "w": w.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Control: random-init transformer (probe should fail)
# ---------------------------------------------------------------------------

def make_random_model(device):
    m = TimeSeriesTransformer(TS_TRANSFORMER_CONFIG).to(device)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ts_transformer.pth")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--freq-min", type=float, default=1.0)
    p.add_argument("--freq-max", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="probes_summary.pdf")
    args = p.parse_args()

    device = torch.device(args.device)

    # --- trained model ---
    model = TimeSeriesTransformer(TS_TRANSFORMER_CONFIG).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint {args.ckpt}")

    ctx_len = TS_TRANSFORMER_CONFIG["context_length"]

    feats_train, f_train = generate_feature_bank(
        model, args.n_train, (args.freq_min, args.freq_max),
        args.fs, ctx_len, seed=args.seed,
    )
    feats_test, f_test = generate_feature_bank(
        model, args.n_test, (args.freq_min, args.freq_max),
        args.fs, ctx_len, seed=args.seed + 999,
    )

    # --- random-init control ---
    random_model = make_random_model(device)
    rfeats_train, rf_train = generate_feature_bank(
        random_model, args.n_train, (args.freq_min, args.freq_max),
        args.fs, ctx_len, seed=args.seed + 7,
    )
    rfeats_test, rf_test = generate_feature_bank(
        random_model, args.n_test, (args.freq_min, args.freq_max),
        args.fs, ctx_len, seed=args.seed + 8,
    )

    # --- fit probes ---
    print("\n=== Trained model ===")
    results = {}
    for name in sorted(feats_train.keys()):
        r = train_linear_probe(
            feats_train[name], f_train, feats_test[name], f_test,
            device=device,
        )
        results[name] = r
        print(f"{name:10s}   R² train {r['r2_train']:.4f}  "
              f"R² test {r['r2_test']:.4f}  "
              f"RMSE test {r['rmse_test']:.3f} Hz")

    print("\n=== Random-init control (should fail) ===")
    random_results = {}
    for name in sorted(rfeats_train.keys()):
        r = train_linear_probe(
            rfeats_train[name], rf_train, rfeats_test[name], rf_test,
            device=device,
        )
        random_results[name] = r
        print(f"{name:10s}   R² train {r['r2_train']:.4f}  "
              f"R² test {r['r2_test']:.4f}  "
              f"RMSE test {r['rmse_test']:.3f} Hz")

    # --- plot ---
    layer_names = sorted(feats_train.keys())
    n = len(layer_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(layer_names):
        # Trained
        ax = axes[0, col]
        r = results[name]
        ax.scatter(f_test, r["pred_test"], s=6, alpha=0.5, color="#065A82")
        lo, hi = args.freq_min, args.freq_max
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("true frequency (Hz)")
        ax.set_ylabel("predicted frequency (Hz)")
        ax.set_title(f"trained  |  {name}\n"
                     f"R² = {r['r2_test']:.3f}   "
                     f"RMSE = {r['rmse_test']:.2f} Hz")
        ax.grid(alpha=0.3)
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_ylim(lo - 1, hi + 1)

        # Random control
        ax = axes[1, col]
        r = random_results[name]
        ax.scatter(rf_test, r["pred_test"], s=6, alpha=0.5, color="#B85042")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("true frequency (Hz)")
        ax.set_ylabel("predicted frequency (Hz)")
        ax.set_title(f"random init  |  {name}\n"
                     f"R² = {r['r2_test']:.3f}   "
                     f"RMSE = {r['rmse_test']:.2f} Hz")
        ax.grid(alpha=0.3)
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_ylim(lo - 1, hi + 1)

    fig.suptitle("Can we read frequency out of the transformer's hidden states?",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"\nSaved {args.out}")

    # --- interpretation ---
    print("\n=== Interpretation ===")
    best_layer = max(results.keys(), key=lambda k: results[k]["r2_test"])
    r = results[best_layer]
    rr = random_results[best_layer]
    print(f"Best trained layer: {best_layer}")
    print(f"  trained  R²={r['r2_test']:.3f}, RMSE={r['rmse_test']:.2f} Hz")
    print(f"  random   R²={rr['r2_test']:.3f}, RMSE={rr['rmse_test']:.2f} Hz")
    if r["r2_test"] - rr["r2_test"] > 0.3:
        print("=> Frequency is encoded in the trained model's hidden state, "
              "not merely the input statistics. The probe works on the trained "
              "model but fails on the random-init control, so training "
              "*caused* the representation to emerge.")


if __name__ == "__main__":
    main()
