"""d2 baseline for stereo-agg-dilation.

Reference: vendor/stereo-matching/baselines/dilation_2.py
"""

_FILE = "stereo-matching/solution/dilation.py"

_CONTENT = 'def build_dilation():\n    return 2'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 22, "end_line": 26, "content": _CONTENT},
]
