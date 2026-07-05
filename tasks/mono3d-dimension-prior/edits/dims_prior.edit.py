"""mono3d-dimension-prior baseline: dims_prior.

Auto-generated from vendor/mono3d-detection/baselines/dims_prior.py. Replaces the editable region of
mono3d-detection/solution/dims_prior.py (the `build_dims_head` surface) with the dims_prior implementation.
"""

_FILE = "mono3d-detection/solution/dims_prior.py"

_CONTENT = 'def build_dims_head(emb_dim, log_mean, cls_dims):\n    lm = log_mean.detach().clone()\n\n    class _Head(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.net = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))\n            self.register_buffer("log_mean", lm)\n\n        def forward(self, emb):\n            return self.net(emb)\n\n    head = _Head()\n\n    def decode(raw, ctx):\n        return torch.exp(head.log_mean.unsqueeze(0) + 0.3 * raw)\n\n    return head, decode'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 38, "content": _CONTENT},
]
