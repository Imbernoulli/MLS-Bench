"""mono3d-depth-normalization baseline: norm_additive.

Auto-generated from vendor/mono3d-detection/baselines/norm_additive.py. Replaces the editable region of
mono3d-detection/solution/depth_norm.py (the `build_depth_norm` surface) with the norm_additive implementation.
"""

_FILE = "mono3d-detection/solution/depth_norm.py"

_CONTENT = 'def build_depth_norm():\n    def apply(geom_Z, raw):\n        return geom_Z + raw[:, 0]            # raw additive metres (badly scaled, can go <=0)\n\n    return apply'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 32, "content": _CONTENT},
]
