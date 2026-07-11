"""Baseline edit for inr-skip-connections/noskip."""

_FILE = "inr-signal-fitting/solution/skip.py"
_CONTENT = '''def surface_config():\n    return {"skip_at": None}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 36, "content": _CONTENT},
]
