"""Good baseline for cv-count-dilation: DILATED large-RF block (CSRNet). Enlarges receptive field without losing resolution -> lower MAE. Ref: vendor/crowd-counting/baselines/dilation_dilated.py"""

_FILE = "crowd-counting/solution/dilation.py"

_CONTENT = '    import torch.nn as nn\n\n    def conv(ci, co, d):\n        return nn.Conv2d(ci, co, 3, padding=d, dilation=d)\n\n    class DilatedBlock(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.net = nn.Sequential(\n                conv(cin, 64, 2), nn.ReLU(True),\n                conv(64, 64, 2), nn.ReLU(True),\n                conv(64, 64, 2), nn.ReLU(True))\n            self.out_channels = 64\n\n        def forward(self, x):\n            return self.net(x)\n\n    return DilatedBlock()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 53, "content": _CONTENT},
]
