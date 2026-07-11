"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/flops_budget.py"

_CONTENT = 'def importance_spec():\n    # Random channel importance under the FLOPs budget (negative control).\n    return {"type": "random"}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
