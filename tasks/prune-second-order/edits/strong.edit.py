"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/second_order.py"

_CONTENT = 'def importance2(name, weight, grad, fisher):\n    # OBS-style second-order importance 0.5 * w^2 * F (Fisher-diagonal proxy).\n    if fisher is None:\n        return weight.abs()\n    return 0.5 * weight.pow(2) * fisher\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
