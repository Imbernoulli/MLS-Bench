"""RMSNorm baseline -- rigorous codebase edit ops.

Replaces CustomSimNorm with group-wise RMSNorm: divides each group
by its root-mean-square, then applies a learnable gain.
Similar structure to SimNorm (groups of simnorm_dim) but uses
RMS normalization instead of softmax.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
"""

_FILE = "tdmpc2/tdmpc2/common/custom_simnorm.py"

_RMSNORM = """\
class CustomSimNorm(nn.Module):
    \"\"\"Group-wise RMSNorm baseline for latent representations.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
        self.eps = 1e-8
        # Learnable gain per group element
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        shp = x.shape
        # Reshape into groups (same as SimNorm)
        x = x.view(*shp[:-1], -1, self.dim)
        # RMS normalization within each group
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=RMSNorm)"
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 16,
        "end_line": 43,
        "content": _RMSNORM,
    },
]
