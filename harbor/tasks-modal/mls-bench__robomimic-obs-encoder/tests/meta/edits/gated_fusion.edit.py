"""Gated fusion encoder baseline -- rigorous codebase edit ops.

Each observation modality is processed by its own MLP, then a learned
sigmoid gate determines how much each modality contributes to the
final representation. This is lighter than attention but more expressive
than simple concatenation, allowing the model to dynamically weight
modality importance.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_obs_encoder.py"

# ── Replace CustomObsEncoder class (lines 19-47) ────────────────────────

_GATED_FUSION = """\
class CustomObsEncoder(nn.Module):
    \"\"\"Gated fusion encoder: sigmoid gates weight each modality.

    Each modality is processed by a small MLP, then a learned gate
    (sigmoid) determines its contribution. The gated features are
    concatenated for the final representation.
    \"\"\"

    def __init__(self, obs_dims, embed_dim=64):
        super().__init__()
        self.obs_dims = obs_dims
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for key in sorted(obs_dims.keys()):
            d = obs_dims[key]
            self.encoders[key] = nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
            )
            self.gates[key] = nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.Sigmoid(),
            )
        self.output_dim = embed_dim * len(obs_dims)

    def forward(self, obs_dict):
        parts = []
        for key in sorted(self.obs_dims.keys()):
            feat = self.encoders[key](obs_dict[key])
            gate = self.gates[key](obs_dict[key])
            parts.append(feat * gate)
        return torch.cat(parts, dim=-1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 19,
        "end_line": 46,
        "content": _GATED_FUSION,
    },
]
