"""L2 Normalization baseline -- rigorous codebase edit ops.

Replaces CustomSimNorm with L2 normalization: projects the latent
vector onto a hypersphere by dividing by its L2 norm.
Unlike SimNorm which constrains to a product of simplices, L2Norm
constrains to a hypersphere, preserving angular relationships.
"""

_FILE = "tdmpc2/tdmpc2/common/custom_simnorm.py"

_L2NORM = """\
class CustomSimNorm(nn.Module):
    \"\"\"L2 normalization baseline -- projects latent vectors onto a hypersphere.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
        self.eps = 1e-8
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # L2 normalize across the last dimension and apply learnable scale
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        return self.scale * x / norm

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=L2Norm)"
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 16,
        "end_line": 43,
        "content": _L2NORM,
    },
]
