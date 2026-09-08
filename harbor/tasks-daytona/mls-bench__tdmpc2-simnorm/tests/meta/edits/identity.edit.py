"""Identity (no normalization) baseline -- rigorous codebase edit ops.

Replaces CustomSimNorm with an identity function that passes the latent
representation through unchanged. Serves as an ablation baseline to
quantify the benefit of normalization in the world model's latent space.
"""

_FILE = "tdmpc2/tdmpc2/common/custom_simnorm.py"

_IDENTITY = """\
class CustomSimNorm(nn.Module):
    \"\"\"Identity baseline -- no normalization applied to latent representations.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        return x

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=Identity)"
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 16,
        "end_line": 43,
        "content": _IDENTITY,
    },
]
