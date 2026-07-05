"""none baseline for stereo-disparity-refinement.

Reference: vendor/stereo-matching/baselines/refine_none.py
"""

_FILE = "stereo-matching/solution/refine.py"

_CONTENT = 'def build_refine():\n    return "none"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 29, "content": _CONTENT},
]
