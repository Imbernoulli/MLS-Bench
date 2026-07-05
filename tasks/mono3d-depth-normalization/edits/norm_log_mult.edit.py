"""mono3d-depth-normalization baseline: norm_log_mult.

Auto-generated from vendor/mono3d-detection/baselines/norm_log_mult.py. Replaces the editable region of
mono3d-detection/solution/depth_norm.py (the `build_depth_norm` surface) with the norm_log_mult implementation.
"""

_FILE = "mono3d-detection/solution/depth_norm.py"

_CONTENT = 'def build_depth_norm():\n    def apply(geom_Z, raw):\n        return geom_Z * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))   # scale-invariant, positive\n\n    return apply'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 32, "content": _CONTENT},
]
