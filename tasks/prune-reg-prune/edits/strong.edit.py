"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/reg_prune.py"

_CONTENT = 'def regularizer(model, params):\n    # L1 weight regularizer that pushes small weights toward zero before\n    # magnitude-thresholding to the enforced budget.\n    l1 = 0.0\n    for name, p in params:\n        l1 = l1 + p.abs().sum()\n    return 1e-5 * l1\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
