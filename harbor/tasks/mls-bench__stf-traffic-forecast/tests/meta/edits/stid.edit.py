"""STID baseline — rigorous codebase edit ops.

Spatial-Temporal Identity (CIKM'22): MLP-based model with learnable
spatial and temporal identity embeddings. No graph structure needed.

Reference: basicts/models/STID/arch/stid_arch.py
"""

_FILE = "BasicTS/custom_model.py"

_CONTENT = """\
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
    hidden_size: int = field(default=32)
    num_layers: int = field(default=3)
    dropout: float = field(default=0.0)


class ResMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))) + x


class Custom(nn.Module):
    \"\"\"STID: Spatial-Temporal Identity baseline.

    Per-node MLP over historical steps, augmented with learnable
    spatial embeddings and temporal (time-of-day, day-of-week) embeddings.
    \"\"\"

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        h = config.hidden_size  # embedding dim for each component

        # Time series embedding: project input_len -> h per node
        self.ts_embed = nn.Linear(config.input_len, h)

        # Spatial embedding: learnable per-node identity [N, h]
        self.spatial_emb = nn.Parameter(torch.empty(config.num_features, h))
        nn.init.xavier_uniform_(self.spatial_emb)

        # Temporal embeddings
        self.tid_emb = nn.Parameter(torch.empty(288, h))  # time-in-day
        nn.init.xavier_uniform_(self.tid_emb)
        self.diw_emb = nn.Parameter(torch.empty(7, h))    # day-in-week
        nn.init.xavier_uniform_(self.diw_emb)

        # Encoder: stack of residual MLPs over concatenated embeddings
        total_h = h * 4  # ts + spatial + tid + diw
        self.encoder = nn.Sequential(
            *[ResMLP(total_h, total_h) for _ in range(config.num_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(total_h, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N], inputs_timestamps: [B, T, 2]
        B, T, N = inputs.shape

        # Time series embedding: [B, N, h]
        ts_emb = self.ts_embed(inputs.transpose(1, 2))

        # Temporal embeddings from last timestamp
        tid_idx = (inputs_timestamps[:, -1, 0] * 288).long()  # [B]
        diw_idx = (inputs_timestamps[:, -1, 1] * 7).long()    # [B]
        tid = self.tid_emb[tid_idx]  # [B, h]
        diw = self.diw_emb[diw_idx]  # [B, h]

        # Expand to [B, N, h]
        spatial = self.spatial_emb.unsqueeze(0).expand(B, -1, -1)
        tid = tid.unsqueeze(1).expand(-1, N, -1)
        diw = diw.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: [B, N, 4*h]
        hidden = torch.cat([ts_emb, spatial, tid, diw], dim=-1)

        # Encode and project: [B, N, output_len] -> [B, output_len, N]
        hidden = self.encoder(hidden)
        prediction = self.output_proj(hidden).transpose(1, 2)
        return prediction
"""

_CONFIG_OVERRIDES = """\
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 2e-3}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 73, "end_line": 75, "content": _CONFIG_OVERRIDES},
    {"op": "replace", "file": _FILE, "start_line": 1, "end_line": 71, "content": _CONTENT},
]
