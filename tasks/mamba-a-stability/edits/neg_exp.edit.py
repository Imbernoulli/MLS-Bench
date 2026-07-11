"""Baseline edit (neg_exp) for the compute_A surface.
Reference: vendor/mamba/baselines/a_neg_exp.py
"""

_FILE = "mamba/solution/a_stability.py"

_CONTENT = '    return {"transform": "neg_exp"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
