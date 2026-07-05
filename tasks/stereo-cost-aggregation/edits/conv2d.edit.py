"""conv2d baseline for stereo-cost-aggregation.

Reference: vendor/stereo-matching/baselines/agg_conv2d.py
"""

_FILE = "stereo-matching/solution/aggregation.py"

_CONTENT = 'def build_aggregation():\n    return "conv2d"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 42, "content": _CONTENT},
]
