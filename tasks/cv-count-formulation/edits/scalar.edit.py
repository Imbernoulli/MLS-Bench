"""Weak baseline (negative control) for cv-count-formulation: direct scalar regression.

Global-average-pool -> MLP -> ONE scalar count. No spatial density: it regresses
toward the (low) TRAIN mean count and cannot extrapolate to the higher-count val
images -> high counting MAE. Reference: vendor/crowd-counting/baselines/head_scalar.py
"""

_FILE = "crowd-counting/solution/head.py"

_CONTENT = '''    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(cin, 128), nn.ReLU(True),
                nn.Linear(128, 64), nn.ReLU(True),
                nn.Linear(64, 1))

        def forward(self, feat):
            pooled = feat.mean(dim=(-2, -1))          # (B, cin) global average pool
            return F.softplus(self.mlp(pooled)).squeeze(-1)   # (B,) scalar count
    return Head()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 46, "end_line": 59, "content": _CONTENT},
]
