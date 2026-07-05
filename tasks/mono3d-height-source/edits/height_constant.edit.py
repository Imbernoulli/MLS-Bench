"""mono3d-height-source baseline: height_constant.

Auto-generated from vendor/mono3d-detection/baselines/height_constant.py. Replaces the editable region of
mono3d-detection/solution/height_source.py (the `build_depth_head` surface) with the height_constant implementation.
"""

_FILE = "mono3d-detection/solution/height_source.py"

_CONTENT = 'def build_depth_head(emb_dim):\n    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))\n\n    def decode(raw, ctx):\n        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)\n        H0 = torch.full_like(h2d, 1.5)                    # GLOBAL constant height for ALL classes\n        geom = ctx["focal"] * H0 / h2d\n        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))\n\n    return head, decode'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 41, "content": _CONTENT},
]
