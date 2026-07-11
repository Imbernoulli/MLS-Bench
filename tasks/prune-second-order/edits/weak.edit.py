"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/second_order.py"

_CONTENT = 'def importance2(name, weight, grad, fisher):\n    # Curvature-free magnitude candidate that ignores the Fisher proxy.\n    return weight.abs()\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
