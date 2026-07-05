"""none baseline for stereo-agg-normalization.

Reference: vendor/stereo-matching/baselines/norm_none.py
"""

_FILE = "stereo-matching/solution/normalization.py"

_CONTENT = 'def build_normalization():\n    return "none"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 23, "end_line": 27, "content": _CONTENT},
]
