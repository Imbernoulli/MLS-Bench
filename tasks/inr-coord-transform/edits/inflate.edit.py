"""Baseline edit for inr-coord-transform/inflate."""

_FILE = "inr-signal-fitting/solution/coord_transform.py"
_CONTENT = '''def surface_config():\n    return {"mode": "inflate", "scale": 100.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
