"""featnorm_l2 baseline for stereo-feature-norm.

Reference: vendor/stereo-matching/baselines/featnorm_l2.py
"""

_FILE = "stereo-matching/solution/featnorm.py"

_CONTENT = 'def build_featnorm():\n    return "l2"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 29, "content": _CONTENT},
]
