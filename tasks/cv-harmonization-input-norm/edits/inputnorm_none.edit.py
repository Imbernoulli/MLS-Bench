"""cv-harmonization-input-norm baseline: none (raw composite input — robust (strong)).
Reference: vendor/image-harmonization/baselines/inputnorm_none.py
"""

_FILE = "image-harmonization/solution/inputnorm.py"

_CONTENT = 'def get_input_norm():\n    return "none"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 21, "end_line": 23, "content": _CONTENT},
]
