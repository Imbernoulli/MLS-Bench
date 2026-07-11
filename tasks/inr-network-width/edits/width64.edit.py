"""Baseline edit for inr-network-width/width64."""

_FILE = "inr-signal-fitting/solution/width.py"
_CONTENT = '''def surface_config():\n    return {"hidden": 64}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
