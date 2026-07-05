"""Weak baseline for cv-count-depth: SHALLOW backbone. Too little capacity for dense crowds -> higher MAE. Ref: vendor/crowd-counting/baselines/depth_shallow.py"""

_FILE = "crowd-counting/solution/depth.py"

_CONTENT = '    import torch.nn as nn\n\n    class Shallow(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.pool = nn.MaxPool2d(2)\n            self.b1 = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(True))\n            self.b2 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(True))\n            self.b3 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(True))\n            self.out_channels = 64\n\n        def forward(self, x):\n            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))\n            return x\n\n    return Shallow()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 55, "content": _CONTENT},
]
