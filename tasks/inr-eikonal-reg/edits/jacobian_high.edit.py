"""Candidate edit for a high RGB Jacobian penalty weight."""

_FILE = "inr-signal-fitting/solution/jacobian_reg.py"
_CONTENT = '''def surface_config():\n    return {"weight": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 36, "content": _CONTENT},
]
