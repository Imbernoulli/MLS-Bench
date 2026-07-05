"""Good baseline for cv-count-upsample: learned UPSAMPLING decoder (finer output). Separates nearby objects in dense scenes -> lower MAE. Ref: vendor/crowd-counting/baselines/upsample_learned.py"""

_FILE = "crowd-counting/solution/upsample.py"

_CONTENT = '    import torch.nn as nn\n\n    class UpDecoder(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.up = nn.ConvTranspose2d(cin, cin, 2, stride=2)\n            self.refine = nn.Sequential(\n                nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(True))\n\n        def forward(self, x):\n            return self.refine(self.up(x))\n\n    return UpDecoder()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 38, "content": _CONTENT},
]
