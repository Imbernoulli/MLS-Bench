"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/structured_criterion.py"

_CONTENT = 'def importance_spec():\n    # L1-norm channel importance (Li et al., ICLR 2017) via Torch-Pruning.\n    return {"type": "l1"}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
