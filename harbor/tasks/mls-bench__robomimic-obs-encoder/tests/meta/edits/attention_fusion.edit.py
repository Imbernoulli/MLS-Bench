"""Attention-based fusion encoder baseline -- rigorous codebase edit ops.

Projects each observation modality to a shared embedding space, then
uses multi-head self-attention to model cross-modality interactions.
This allows the encoder to learn which observation modalities are most
relevant and how they relate to each other.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_obs_encoder.py"

# ── Replace CustomObsEncoder class (lines 19-47) ────────────────────────

_ATTENTION_FUSION = """\
class CustomObsEncoder(nn.Module):
    \"\"\"Attention-based cross-modality fusion encoder.

    Projects each modality to a shared embedding space, then applies
    multi-head self-attention across modalities. The attended features
    are concatenated for the final representation.
    \"\"\"

    def __init__(self, obs_dims, embed_dim=64, num_heads=2):
        super().__init__()
        self.obs_dims = obs_dims
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict()
        for key in sorted(obs_dims.keys()):
            d = obs_dims[key]
            self.projections[key] = nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.ReLU(),
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim * len(obs_dims)

    def forward(self, obs_dict):
        tokens = []
        for key in sorted(self.obs_dims.keys()):
            tokens.append(self.projections[key](obs_dict[key]))
        tokens = torch.stack(tokens, dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.reshape(tokens.shape[0], -1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 19,
        "end_line": 46,
        "content": _ATTENTION_FUSION,
    },
]
