"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/reg_prune.py"

_CONTENT = 'def regularizer(model, params):\n    # NO sparsity regularizer (negative control) -> plain magnitude thresholding.\n    return 0.0\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
