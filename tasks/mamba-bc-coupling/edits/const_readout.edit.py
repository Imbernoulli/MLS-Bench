"""Baseline edit (const_readout) for the couple_bc surface.
Reference: vendor/mamba/baselines/bc_const_readout.py
"""

_FILE = "mamba/solution/bc_coupling.py"

_CONTENT = '    return {"coupling": "constant"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
