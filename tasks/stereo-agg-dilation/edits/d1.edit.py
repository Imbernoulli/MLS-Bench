"""d1 baseline for stereo-agg-dilation.

Reference: vendor/stereo-matching/baselines/dilation_1.py
"""

_FILE = "stereo-matching/solution/dilation.py"

_CONTENT = 'def build_dilation():\n    return 1'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 22, "end_line": 26, "content": _CONTENT},
]
