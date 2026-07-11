"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/layer_budget.py"

_CONTENT = 'def layer_sparsity(layer_names):\n    # UNIFORM: every layer pruned to the global target (no sensitivity awareness).\n    return {}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
