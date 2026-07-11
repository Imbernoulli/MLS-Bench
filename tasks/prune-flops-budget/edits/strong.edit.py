"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/flops_budget.py"

_CONTENT = 'def importance_spec():\n    # L1 channel-importance candidate under the fixed measured-MAC budget.\n    return {"type": "l1"}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
