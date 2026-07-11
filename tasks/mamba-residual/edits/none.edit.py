"""Baseline edit (none) for the residual_step surface.
Reference: vendor/mamba/baselines/resid_none.py
"""

_FILE = "mamba/solution/residual.py"

_CONTENT = '    return {"residual": "none"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
