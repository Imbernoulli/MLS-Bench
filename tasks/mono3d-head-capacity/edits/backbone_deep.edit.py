"""mono3d-head-capacity baseline: backbone_deep.

Auto-generated from vendor/mono3d-detection/baselines/backbone_deep.py. Replaces the editable region of
mono3d-detection/solution/backbone.py (the `build_backbone` surface) with the backbone_deep implementation.
"""

_FILE = "mono3d-detection/solution/backbone.py"

_CONTENT = 'def build_backbone(emb_dim):\n    class _Block(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.f = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),\n                                   nn.Linear(2 * emb_dim, emb_dim))\n\n        def forward(self, x):\n            return F.relu(x + self.f(x))     # residual: preserves info + adds capacity\n\n    class _Deep(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.b1 = _Block()\n            self.b2 = _Block()\n\n        def forward(self, x):\n            return self.b2(self.b1(x))\n\n    return _Deep()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 28, "end_line": 39, "content": _CONTENT},
]
