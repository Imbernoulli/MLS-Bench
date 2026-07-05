"""Weak baseline for cv-count-dilation: POOLED small-RF block. Loses resolution / context -> higher MAE. Ref: vendor/crowd-counting/baselines/dilation_pooled.py"""

_FILE = "crowd-counting/solution/dilation.py"

_CONTENT = '    import torch.nn as nn\n\n    class PooledBlock(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.net = nn.Sequential(\n                nn.MaxPool2d(2),\n                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),\n                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),\n                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))\n            self.out_channels = 64\n\n        def forward(self, x):\n            return self.net(x)\n\n    return PooledBlock()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 53, "content": _CONTENT},
]
