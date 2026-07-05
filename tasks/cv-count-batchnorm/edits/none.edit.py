"""Weak baseline for cv-count-batchnorm: NO normalization. Activation stats drift -> higher MAE. Ref: vendor/crowd-counting/baselines/batchnorm_none.py"""

_FILE = "crowd-counting/solution/batchnorm.py"

_CONTENT = '    import torch.nn as nn\n\n    def conv(cin, cout, d=1):\n        return nn.Conv2d(cin, cout, 3, padding=d, dilation=d)\n\n    class PlainBackbone(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.pool = nn.MaxPool2d(2)\n            self.b1 = nn.Sequential(conv(3, 32), nn.ReLU(True), conv(32, 32), nn.ReLU(True))\n            self.b2 = nn.Sequential(conv(32, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))\n            self.b3 = nn.Sequential(conv(64, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))\n            self.out_channels = 64\n\n        def forward(self, x):\n            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))\n            return x\n\n    return PlainBackbone()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 60, "content": _CONTENT},
]
