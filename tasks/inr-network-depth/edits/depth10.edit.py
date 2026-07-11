"""Baseline edit for inr-network-depth/depth10."""

_FILE = "inr-signal-fitting/solution/depth.py"
_CONTENT = '''def surface_config():\n    return {"n_layers": 10}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 7, "end_line": 8, "content": _CONTENT},
]
