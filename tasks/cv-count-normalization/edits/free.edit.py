"""Good baseline for cv-count-normalization: free non-negative density field.

Per-pixel softplus density with UNBOUNDED total mass -> the integral scales to any
count -> it extrapolates to the higher-count val images -> lower counting MAE with
clear headroom. Reference: vendor/crowd-counting/baselines/norm_free.py
"""

_FILE = "crowd-counting/solution/norm.py"

_CONTENT = '''    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))

        def forward(self, feat):
            return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) free density
    return Head()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 56, "end_line": 71, "content": _CONTENT},
]
