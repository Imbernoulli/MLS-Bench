"""Good baseline for cv-count-attention: spatial ATTENTION gate. Suppresses distractor clutter -> lower MAE. Ref: vendor/crowd-counting/baselines/attention_spatial.py"""

_FILE = "crowd-counting/solution/attention.py"

_CONTENT = '    import torch.nn as nn\n\n    class SpatialAttention(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.gate = nn.Sequential(\n                nn.Conv2d(cin, cin // 2, 3, padding=1), nn.ReLU(True),\n                nn.Conv2d(cin // 2, 1, 1), nn.Sigmoid())\n\n        def forward(self, x):\n            return x * self.gate(x)   # per-pixel gated features\n\n    return SpatialAttention()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
