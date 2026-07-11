"""Baseline edit for inr-activation/relu_mlp."""

_FILE = "inr-signal-fitting/solution/activation.py"
_CONTENT = '''def surface_config():\n    return {"family": "relu"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 40, "content": _CONTENT},
]
