"""cv-harmonization-activation baseline: identity (no nonlinearity — collapses to linear (weak)).
Reference: vendor/image-harmonization/baselines/activation_identity.py
"""

_FILE = "image-harmonization/solution/activation.py"

_CONTENT = 'def get_activation():\n    return "identity"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 18, "content": _CONTENT},
]
