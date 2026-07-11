"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/reinit.py"

_CONTENT = 'def reinit():\n    # Keep the trained surviving weights and recover them in place.\n    return "keep"\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
