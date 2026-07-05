"""mono3d-projection baseline: proj_onaxis.

Auto-generated from vendor/mono3d-detection/baselines/proj_onaxis.py. Replaces the editable region of
mono3d-detection/solution/projection.py (the `build_backproject` surface) with the proj_onaxis implementation.
"""

_FILE = "mono3d-detection/solution/projection.py"

_CONTENT = 'def build_backproject():\n    def backproject(loc_z, box2d, cx, cy, focal):\n        z = loc_z.reshape(-1)\n        x = torch.zeros_like(z)          # ON-AXIS: ignore the 2D-box horizontal position\n        y = torch.zeros_like(z)          # ON-AXIS: ignore the 2D-box vertical position\n        return x, y\n\n    return backproject'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 36, "content": _CONTENT},
]
