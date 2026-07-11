"""Baseline edit for inr-network-width/width8."""

_FILE = "inr-signal-fitting/solution/width.py"
_CONTENT = '''def surface_config():\n    return {"hidden": 8}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
