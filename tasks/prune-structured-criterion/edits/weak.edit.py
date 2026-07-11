"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/structured_criterion.py"

_CONTENT = 'def importance_spec():\n    # Random channel importance (negative control).\n    return {"type": "random"}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
