"""Baseline edit (tied) for the couple_bc surface.
Reference: vendor/mamba/baselines/bc_tied.py
"""

_FILE = "mamba/solution/bc_coupling.py"

_CONTENT = '    return {"coupling": "tied"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
