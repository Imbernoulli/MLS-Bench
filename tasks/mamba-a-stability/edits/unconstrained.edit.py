"""Baseline edit (unconstrained) for the compute_A surface.
Reference: vendor/mamba/baselines/a_unconstrained.py
"""

_FILE = "mamba/solution/a_stability.py"

_CONTENT = '    return {"transform": "identity"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
