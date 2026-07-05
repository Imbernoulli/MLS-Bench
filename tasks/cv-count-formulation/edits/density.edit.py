"""Good baseline for cv-count-formulation: density-map integral.

Predict a NON-NEGATIVE per-pixel density map and count by its spatial integral (the
MCNN / CSRNet formulation): translation-equivariant local density that GENERALISES to
the higher-count val images -> low counting MAE with large headroom over direct
scalar regression. Reference: vendor/crowd-counting/baselines/head_density.py
"""

_FILE = "crowd-counting/solution/head.py"

_CONTENT = '''    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))

        def forward(self, feat):
            return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) density
    return Head()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 46, "end_line": 59, "content": _CONTENT},
]
