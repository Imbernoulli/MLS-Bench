"""SOFTS baseline — rigorous codebase edit ops.

SOFTS (NeurIPS'24): Series-Core Fusion via STar
Aggregate-Redistribute (STAR). Inverted view (like iTransformer)
with STAR replacing self-attention for efficient cross-variate modeling.

Reference: basicts/models/SOFTS/arch/softs_arch.py
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
    core_size: int = field(default=128)
    num_layers: int = field(default=2)
    dropout: float = field(default=0.05)


class RevIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:
            return x * self.stdev + self.mean


class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class STAR(nn.Module):
    \"\"\"STar Aggregate-Redistribute module.

    Aggregates cross-variate info into a core representation via
    stochastic pooling (training) or weighted mean (inference),
    then redistributes back to each variate.
    \"\"\"
    def __init__(self, hidden_size, core_size):
        super().__init__()
        self.ffn1 = MLP(hidden_size, hidden_size, core_size)
        self.ffn2 = MLP(hidden_size + core_size, hidden_size, hidden_size)

    def forward(self, x):
        B, N, D = x.shape
        combined = self.ffn1(x)  # [B, N, core_size]

        if self.training:
            # Stochastic pooling
            ratio = F.softmax(combined, dim=1)  # [B, N, core_size]
            ratio = ratio.transpose(1, 2).reshape(-1, N)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(B, -1, 1).transpose(1, 2)  # [B, 1, core_size]
            core = torch.gather(combined, 1, indices)  # [B, 1, core_size]
            core = core.repeat(1, N, 1)
        else:
            # Weighted mean
            weight = F.softmax(combined, dim=1)
            core = (combined * weight).sum(dim=1, keepdim=True).repeat(1, N, 1)

        return self.ffn2(torch.cat([x, core], dim=-1))


class SOFTSBlock(nn.Module):
    def __init__(self, hidden_size, core_size, dropout):
        super().__init__()
        self.star = STAR(hidden_size, core_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.star(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Custom(nn.Module):
    \"\"\"SOFTS: Series-Core Fusion baseline.

    Inverted architecture (nodes as tokens), using STAR modules
    instead of self-attention for O(N) cross-variate communication.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.revin = RevIN()

        # Sequence embedding: [B, T, N] -> transpose -> [B, N, T] -> [B, N, D]
        self.embed = nn.Linear(config.input_len, config.hidden_size)
        self.embed_drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            SOFTSBlock(config.hidden_size, config.core_size, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output: [B, N, D] -> [B, N, T'] -> [B, T', N]
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        x = self.revin(inputs, "norm")
        N = x.size(-1)

        h = self.embed_drop(self.embed(x.transpose(1, 2)))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        pred = self.head(h).transpose(1, 2)[:, :, :N]
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
