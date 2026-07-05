"""featnorm_none baseline for stereo-feature-norm.

Reference: vendor/stereo-matching/baselines/featnorm_none.py
"""

_FILE = "stereo-matching/solution/featnorm.py"

_CONTENT = 'def build_featnorm():\n    return "none"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 29, "content": _CONTENT},
]
