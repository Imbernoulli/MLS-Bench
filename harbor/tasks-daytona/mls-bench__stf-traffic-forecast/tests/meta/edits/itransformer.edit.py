"""iTransformer baseline — rigorous codebase edit ops.

iTransformer (ICLR'24): Inverted Transformer that applies attention
over the feature (node) dimension instead of the time dimension.
Each node's time series is embedded as a token.

Reference: basicts/models/iTransformer/arch/itransformer_arch.py
"""

_FILE = "BasicTS/custom_model.py"

_CONTENT = """\
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=512)
    n_heads: int = field(default=8)
    num_layers: int = field(default=3)
    dropout: float = field(default=0.1)


class RevIN(nn.Module):
    \"\"\"Reversible Instance Normalization.\"\"\"
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:  # denorm
            return x * self.stdev + self.mean


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Custom(nn.Module):
    \"\"\"iTransformer: Inverted Transformer baseline.

    Treats each node's time series as a token (inverted view).
    Self-attention captures cross-variate dependencies.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.num_features = config.num_features
        self.revin = RevIN()

        # Embed each node's input_len time series -> hidden_size
        self.embed = nn.Linear(config.input_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder over nodes
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.n_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Project hidden_size -> output_len per node
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        x = self.revin(inputs, "norm")

        # Invert: [B, N, T] -> embed -> [B, N, D]
        h = self.dropout(self.embed(x.transpose(1, 2)))

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Project: [B, N, D] -> [B, N, T'] -> [B, T', N]
        pred = self.head(h).transpose(1, 2)[:, :, :self.num_features]
        pred = self.revin(pred, "denorm")
        return pred
"""

_CONFIG_OVERRIDES = """\
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.0005}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 73, "end_line": 75, "content": _CONFIG_OVERRIDES},
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 71, "content": _CONTENT},
]
