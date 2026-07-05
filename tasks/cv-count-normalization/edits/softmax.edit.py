"""Weak baseline (negative control) for cv-count-normalization: softmax x scalar.

Spatial softmax (mass=1) times a single learned count scalar; the scalar saturates at
the low training mean and cannot scale up to the higher-count val images -> high
counting MAE. Reference: vendor/crowd-counting/baselines/norm_softmax.py
"""

_FILE = "crowd-counting/solution/norm.py"

_CONTENT = '''    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))
            self.count_scalar = nn.Parameter(torch.tensor(50.0))

        def forward(self, feat):
            m = self.net(feat)                       # (B,1,h,w)
            B, _, h, w = m.shape
            dist = F.softmax(m.view(B, -1), dim=1).view(B, h, w)   # sums to 1 per image
            total = F.softplus(self.count_scalar) * 100.0          # DENSITY_SCALE=100
            return dist * total                      # (B,h,w) density, mass = total
    return Head()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 56, "end_line": 71, "content": _CONTENT},
]
