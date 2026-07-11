"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/norm.py"

_CONTENT = '''    class RegionNorm(nn.Module):
        def __init__(self, num_ch, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(1, num_ch, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_ch, 1, 1))

        def forward(self, x):
            b, c, h, w = x.shape
            energy = x.abs().mean(dim=1, keepdim=True)                 # (B,1,H,W)
            med = energy.flatten(2).median(dim=-1, keepdim=True)[0].unsqueeze(-1)
            valid = torch.sigmoid((energy - 0.5 * med) * 20.0)         # soft in [0,1]
            vsum = valid.sum(dim=(2, 3), keepdim=True) + self.eps
            mean = (x * valid).sum(dim=(2, 3), keepdim=True) / vsum
            var = ((x - mean) ** 2 * valid).sum(dim=(2, 3), keepdim=True) / vsum
            xn = (x - mean) / torch.sqrt(var + self.eps)
            return xn * self.gamma + self.beta

    return RegionNorm(num_ch)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 17, "content": _CONTENT},
]
