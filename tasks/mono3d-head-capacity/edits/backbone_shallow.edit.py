"""mono3d-head-capacity baseline: backbone_shallow.

Auto-generated from vendor/mono3d-detection/baselines/backbone_shallow.py. Replaces the editable region of
mono3d-detection/solution/backbone.py (the `build_backbone` surface) with the backbone_shallow implementation.
"""

_FILE = "mono3d-detection/solution/backbone.py"

_CONTENT = 'def build_backbone(emb_dim):\n    class _Narrow(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.f = nn.Sequential(nn.Linear(emb_dim, 8), nn.ReLU(), nn.Linear(8, emb_dim))\n\n        def forward(self, x):\n            return self.f(x)                 # NO residual: the 8-d bottleneck is a hard squeeze\n\n    return _Narrow()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 28, "end_line": 39, "content": _CONTENT},
]
