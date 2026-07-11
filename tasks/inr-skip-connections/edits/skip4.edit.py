"""Baseline edit for inr-skip-connections/skip4."""

_FILE = "inr-signal-fitting/solution/skip.py"
_CONTENT = '''def surface_config():\n    return {"skip_at": 4}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 36, "content": _CONTENT},
]
