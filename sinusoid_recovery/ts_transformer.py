import torch
import torch.nn as nn

from transformer_blocks import MultiHeadAttention, TransformerBlock, LayerNorm, GELU, FeedForward


TS_TRANSFORMER_CONFIG = {
    "context_length": 128,   # covers ~1 full period of lowest freq (1 Hz at fs=100)
    "emb_dim": 32,           # small: each token is just a scalar sample
    "n_heads": 2,            # 2 heads can learn lag-1 and lag-2 patterns
    "n_layers": 2,           # minimal depth
    "drop_rate": 0.05,       # light dropout
    "qkv_bias": False,
}


class TimeSeriesTransformer(nn.Module):
    """Transformer for next-sample prediction on continuous time series.

    Adapted from GPTModel in ch04/01_main-chapter-code/gpt.py:
    - Token embedding (nn.Embedding) -> Linear projection (nn.Linear(1, emb_dim))
    - Output head -> nn.Linear(emb_dim, 1) for scalar prediction
    - Everything else (attention, FFN, layer norm, positional embeddings) is reused.
    """

    def __init__(self, cfg):
        super().__init__()
        # Project scalar sample value into embedding space
        self.input_proj = nn.Linear(1, cfg["emb_dim"])
        # Learnable positional embeddings (position encodes time/lag information)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Reuse TransformerBlock from the GPT implementation
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Output: predict a single scalar (next sample value)
        self.out_head = nn.Linear(cfg["emb_dim"], 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - sequence of scalar sample values

        Returns:
            predictions: (batch_size, seq_len) - predicted next sample at each position
        """
        batch_size, seq_len = x.shape

        # Reshape to (batch, seq_len, 1) for linear projection
        x = x.unsqueeze(-1)
        # Project scalar -> emb_dim
        tok_embeds = self.input_proj(x)  # (batch, seq_len, emb_dim)
        # Add positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Transformer blocks (with causal masking from MultiHeadAttention)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Project back to scalar prediction
        predictions = self.out_head(x).squeeze(-1)  # (batch, seq_len)
        return predictions


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(42)
    model = TimeSeriesTransformer(TS_TRANSFORMER_CONFIG)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Config: {TS_TRANSFORMER_CONFIG}")

    # Test forward pass
    x = torch.randn(4, TS_TRANSFORMER_CONFIG["context_length"])  # batch of 4 sequences
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
