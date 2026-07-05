"""mono3d-depth-cue baseline: cue_height.

Auto-generated from vendor/mono3d-detection/baselines/cue_height.py. Replaces the editable region of
mono3d-detection/solution/depth_cue.py (the `build_depth_head` surface) with the cue_height implementation.
"""

_FILE = "mono3d-detection/solution/depth_cue.py"

_CONTENT = 'def build_depth_head(emb_dim):\n    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))\n\n    def decode(raw, ctx):\n        H = ctx["pred_H"].reshape(-1)\n        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)\n        geom = ctx["focal"] * H / h2d\n        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))\n\n    return head, decode'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 42, "content": _CONTENT},
]
