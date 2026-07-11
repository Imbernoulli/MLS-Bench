"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/criterion.py"

_CONTENT = 'def importance(name, weight, grad):\n    # RANDOM importance -> an arbitrary subnetwork survives (negative control).\n    return torch.rand_like(weight)\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
