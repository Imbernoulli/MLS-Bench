"""Baseline edit for variance-preserving residual addition."""

_FILE = "mamba/solution/residual.py"
_CONTENT = '    return {"residual": "scaled_add"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
