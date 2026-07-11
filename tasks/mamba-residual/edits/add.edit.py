"""Baseline edit (add) for the residual_step surface.
Reference: vendor/mamba/baselines/resid_add.py
"""

_FILE = "mamba/solution/residual.py"

_CONTENT = '    return {"residual": "add"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
