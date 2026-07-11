"""Baseline edit for inr-per-layer-w0/w0_15."""

_FILE = "inr-signal-fitting/solution/per_layer_w0.py"
_CONTENT = '''def surface_config():\n    return {"first": 15.0, "hidden": 15.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
