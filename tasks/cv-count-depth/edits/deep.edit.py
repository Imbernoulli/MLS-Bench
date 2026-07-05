"""Good baseline for cv-count-depth: DEEP backbone. More capacity to resolve heavily crowded scenes -> lower MAE. Ref: vendor/crowd-counting/baselines/depth_deep.py"""

_FILE = "crowd-counting/solution/depth.py"

_CONTENT = '    import torch.nn as nn\n\n    def cbr(ci, co):\n        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(True))\n\n    class Deep(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.pool = nn.MaxPool2d(2)\n            self.b1 = nn.Sequential(cbr(3, 32), cbr(32, 32))\n            self.b2 = nn.Sequential(cbr(32, 64), cbr(64, 64))\n            self.b3 = nn.Sequential(cbr(64, 64), cbr(64, 64))\n            self.refine = nn.Sequential(cbr(64, 64), cbr(64, 64))\n            self.out_channels = 64\n\n        def forward(self, x):\n            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))\n            return self.refine(x)\n\n    return Deep()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 55, "content": _CONTENT},
]
