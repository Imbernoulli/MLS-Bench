"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/schedule.py"

_CONTENT = 'def schedule(target_sparsity, total_steps):\n    # GRADUAL / iterative magnitude pruning across 3 rungs (Zhu & Gupta, 2017).\n    third = max(1, total_steps // 3)\n    return [(target_sparsity * 0.34, third),\n            (target_sparsity * 0.67, third),\n            (target_sparsity, max(1, total_steps - 2 * third))]\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
