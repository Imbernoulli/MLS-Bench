"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/schedule.py"

_CONTENT = 'def schedule(target_sparsity, total_steps):\n    # ONE-SHOT: jump straight to the target, then spend the whole budget recovering.\n    return [(target_sparsity, total_steps)]\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
