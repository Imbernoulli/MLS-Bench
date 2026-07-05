"""mono3d-projection baseline: proj_pinhole.

Auto-generated from vendor/mono3d-detection/baselines/proj_pinhole.py. Replaces the editable region of
mono3d-detection/solution/projection.py (the `build_backproject` surface) with the proj_pinhole implementation.
"""

_FILE = "mono3d-detection/solution/projection.py"

_CONTENT = 'def build_backproject():\n    def backproject(loc_z, box2d, cx, cy, focal):\n        u = 0.5 * (box2d[:, 0] + box2d[:, 2])\n        v = 0.5 * (box2d[:, 1] + box2d[:, 3])\n        z = loc_z.reshape(-1)\n        x = (u - cx) * z / focal\n        y = (v - cy) * z / focal\n        return x, y\n\n    return backproject'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 36, "content": _CONTENT},
]
