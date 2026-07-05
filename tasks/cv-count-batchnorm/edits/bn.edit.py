"""Good baseline for cv-count-batchnorm: BatchNorm backbone. Stabler optimisation across the count range -> lower MAE. Ref: vendor/crowd-counting/baselines/batchnorm_bn.py"""

_FILE = "crowd-counting/solution/batchnorm.py"

_CONTENT = '    import torch.nn as nn\n\n    def cbr(cin, cout):\n        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),\n                             nn.BatchNorm2d(cout), nn.ReLU(True))\n\n    class BNBackbone(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.pool = nn.MaxPool2d(2)\n            self.b1 = nn.Sequential(cbr(3, 32), cbr(32, 32))\n            self.b2 = nn.Sequential(cbr(32, 64), cbr(64, 64))\n            self.b3 = nn.Sequential(cbr(64, 64), cbr(64, 64))\n            self.out_channels = 64\n\n        def forward(self, x):\n            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))\n            return x\n\n    return BNBackbone()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 60, "content": _CONTENT},
]
