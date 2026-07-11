"""Baseline edit for inr-per-layer-w0/w0_3."""

_FILE = "inr-signal-fitting/solution/per_layer_w0.py"
_CONTENT = '''def surface_config():\n    return {"first": 3.0, "hidden": 3.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
