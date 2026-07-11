"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/reinit.py"

_CONTENT = 'def reinit():\n    # Randomly RE-INITIALIZE the surviving weights (negative control).\n    return "random"\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
