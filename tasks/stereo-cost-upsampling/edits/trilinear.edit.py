"""trilinear baseline for stereo-cost-upsampling.

Reference: vendor/stereo-matching/baselines/upsample_trilinear.py
"""

_FILE = "stereo-matching/solution/upsampling.py"

_CONTENT = 'def build_upsampling():\n    return "trilinear"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 29, "content": _CONTENT},
]
