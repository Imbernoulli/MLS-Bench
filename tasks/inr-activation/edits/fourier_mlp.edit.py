"""Baseline edit for inr-activation/fourier_mlp."""

_FILE = "inr-signal-fitting/solution/activation.py"
_CONTENT = '''def surface_config():\n    return {"family": "fourier"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 40, "content": _CONTENT},
]
