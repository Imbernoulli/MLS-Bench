"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/taylor_estimator.py"

_CONTENT = 'def estimate_importance(model, batches, params):\n    # Data-free magnitude candidate that ignores the supplied calibration pass.\n    return {name: p.detach().abs() for name, p in params}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
