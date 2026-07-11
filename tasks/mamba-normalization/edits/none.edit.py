"""Baseline edit (none) for the make_norm surface.
Reference: vendor/mamba/baselines/norm_none.py
"""

_FILE = "mamba/solution/normalization.py"

_CONTENT = '    return {"normalization": "none"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
