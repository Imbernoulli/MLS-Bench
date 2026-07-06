"""cv-harmonization-activation baseline: gelu (GELU — strong).
Reference: vendor/image-harmonization/baselines/activation_gelu.py
"""

_FILE = "image-harmonization/solution/activation.py"

_CONTENT = 'def get_activation():\n    return "gelu"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 18, "content": _CONTENT},
]
