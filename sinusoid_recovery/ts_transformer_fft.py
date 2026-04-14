"""Time-series transformer with a sliding-window FFT front-end.

Replaces the scalar Linear(1, emb_dim) embedding in ts_transformer.py with
a causal sliding-window FFT: each token carries local frequency content.

For position t:
    token(t) = Linear( rfft( x[t-W+1 : t+1] ) )
with left-padding of W-1 zeros to keep the sequence length at context_length.
"""

import torch
import torch.nn as nn

from transformer_blocks import TransformerBlock, LayerNorm


TS_TRANSFORMER_FFT_CONFIG = {
    "context_length": 128,
    "window_size": 32,        # sliding FFT window length
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.05,
    "qkv_bias": False,
}


class FFTEmbedding(nn.Module):
    """Causal sliding-window FFT embedding.

    Input:  (B, T) real samples
    Output: (B, T, emb_dim) — FFT features projected to emb_dim
    """

    def __init__(self, window_size, emb_dim):
        super().__init__()
        self.W = window_size
        # rfft of W real samples -> W//2 + 1 complex bins -> 2*(W//2+1) real features
        n_feats = 2 * (window_size // 2 + 1)
        self.n_feats = n_feats
        self.proj = nn.Linear(n_feats, emb_dim)

    def forward(self, x):
        B, T = x.shape
        # Left-pad with W-1 zeros to preserve causality and sequence length
        pad = torch.zeros(B, self.W - 1, device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)                  # (B, T + W - 1)
        # Sliding windows of length W at every position: (B, T, W)
        windows = x_pad.unfold(dimension=1, size=self.W, step=1)
        # FFT along the window dimension -> (B, T, W//2+1) complex
        spec = torch.fft.rfft(windows, dim=-1)
        # Stack real/imag -> (B, T, 2*(W//2+1))
        feats = torch.cat([spec.real, spec.imag], dim=-1)
        return self.proj(feats)


class TimeSeriesTransformerFFT(nn.Module):
    """Same architecture as TimeSeriesTransformer but with FFT-based embedding."""

    def __init__(self, cfg):
        super().__init__()
        self.fft_emb = FFTEmbedding(cfg["window_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], 1)

    def forward(self, x):
        B, T = x.shape
        tok_embeds = self.fft_emb(x)                              # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(T, device=x.device))
        h = tok_embeds + pos_embeds
        h = self.drop_emb(h)
        h = self.trf_blocks(h)
        h = self.final_norm(h)
        return self.out_head(h).squeeze(-1)                       # (B, T)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TimeSeriesTransformerFFT(TS_TRANSFORMER_FFT_CONFIG)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Config: {TS_TRANSFORMER_FFT_CONFIG}")
    x = torch.randn(4, TS_TRANSFORMER_FFT_CONFIG["context_length"])
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
