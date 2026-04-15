"""Step-4a learned receiver: a small transformer over the post-FFT grid.

Input:   (B, 14, 300, 3)   real features per (symbol, subcarrier)
Output:  (B, 14, 300, 2)   logits for bit0 and bit1 at every cell

We compute the loss only on data cells (pilots are masked out in train.py).

Design choices:
    d_model=128, 4 heads (head_dim=32), 4 layers, d_ff=512.
    Factorized 2-D positional embedding: one learned embedding per symbol
    index (14) and one per subcarrier index (300), summed into d_model.
    That costs 14*128 + 300*128 = 40k params vs 14*300*128 = 537k for dense.

Parameter count: ~840k. Fits on any GPU; runs on CPU for smoke tests.
"""

import torch
from torch import nn


class FactorizedPositionalEmbedding(nn.Module):
    def __init__(self, n_symbols: int, n_subcarriers: int, d_model: int):
        super().__init__()
        self.sym_emb = nn.Embedding(n_symbols, d_model)
        self.sc_emb = nn.Embedding(n_subcarriers, d_model)
        self.n_symbols = n_symbols
        self.n_subcarriers = n_subcarriers

    def forward(self, x):
        # x: (B, T, S, d)
        B, T, S, D = x.shape
        sym_ids = torch.arange(T, device=x.device)
        sc_ids = torch.arange(S, device=x.device)
        sym = self.sym_emb(sym_ids)[None, :, None, :]   # (1,T,1,D)
        sc = self.sc_emb(sc_ids)[None, None, :, :]      # (1,1,S,D)
        return x + sym + sc


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, d)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.ln2(x)
        x = x + self.drop(self.ff(h))
        return x


class LearnedReceiver(nn.Module):
    def __init__(self,
                 n_symbols: int = 14,
                 n_subcarriers: int = 300,
                 in_features: int = 3,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.pos = FactorizedPositionalEmbedding(n_symbols, n_subcarriers, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)   # 2 bit logits per cell
        self.n_symbols = n_symbols
        self.n_subcarriers = n_subcarriers

    def forward(self, rx_grid):
        # rx_grid: (B, T, S, in_features)
        B, T, S, _ = rx_grid.shape
        x = self.in_proj(rx_grid)                 # (B, T, S, d)
        x = self.pos(x)
        x = x.view(B, T * S, -1)                  # flatten to token sequence
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        logits = self.head(x)                     # (B, T*S, 2)
        return logits.view(B, T, S, 2)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
