"""Transformer mapping-network baseline (ClipCap transformer variant).

Instead of expanding the CLIP vector into the whole prefix with one MLP, project
the CLIP vector to `gpt_dim`, treat it as a single visual token, and let a small
transformer refine `prefix_len` learned constant queries that cross-attend to it.
Each prefix token can then specialise, which ClipCap reports gives more grounded
captions than the flat MLP prefix. Reference: Mokady et al. 2021, ClipCap
(transformer mapping network).
"""
import torch
import torch.nn as nn


class _TransformerMapping(nn.Module):
    def __init__(self, clip_dim, gpt_dim, prefix_len, n_layers=4, n_heads=8):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.proj = nn.Linear(clip_dim, gpt_dim)
        self.queries = nn.Parameter(torch.randn(prefix_len, gpt_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=gpt_dim, nhead=n_heads, dim_feedforward=gpt_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        B = x.shape[0]
        vis = self.proj(x).unsqueeze(1)                          # [B,1,D]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # [B,P,D]
        seq = torch.cat([vis, q], dim=1)                         # [B,1+P,D]
        out = self.enc(seq)
        return out[:, 1:, :]                                     # [B,P,D]


def build_mapping(clip_dim, gpt_dim, prefix_len):
    return _TransformerMapping(clip_dim, gpt_dim, prefix_len)
