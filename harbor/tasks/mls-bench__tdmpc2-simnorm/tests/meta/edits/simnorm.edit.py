"""SimNorm (original) baseline -- rigorous codebase edit ops.

Keeps the default SimNorm (Simplicial Normalization) as a baseline.
SimNorm reshapes the latent vector into groups of simnorm_dim and
applies softmax within each group, constraining each group to lie
on a simplex.

Reference: TD-MPC2 (Hansen et al., ICLR 2024).
"""

_FILE = "tdmpc2/tdmpc2/common/custom_simnorm.py"

_SIMNORM = """\
class CustomSimNorm(nn.Module):
    \"\"\"SimNorm baseline -- original simplicial normalization from TD-MPC2.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=SimNorm)"
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 16,
        "end_line": 43,
        "content": _SIMNORM,
    },
]
