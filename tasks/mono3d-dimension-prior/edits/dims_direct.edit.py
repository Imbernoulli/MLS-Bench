"""mono3d-dimension-prior baseline: dims_direct.

Auto-generated from vendor/mono3d-detection/baselines/dims_direct.py. Replaces the editable region of
mono3d-detection/solution/dims_prior.py (the `build_dims_head` surface) with the dims_direct implementation.
"""

_FILE = "mono3d-detection/solution/dims_prior.py"

_CONTENT = 'def build_dims_head(emb_dim, log_mean, cls_dims):\n    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))\n\n    def decode(raw, ctx):\n        # DIRECT positive dims via softplus, ignoring the class-mean prior entirely.\n        return torch.nn.functional.softplus(raw) + 0.05\n\n    return head, decode'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 38, "content": _CONTENT},
]
