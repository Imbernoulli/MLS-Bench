"""Baseline edit (none) for the gate surface.
Reference: vendor/mamba/baselines/gate_none.py
"""

_FILE = "mamba/solution/gating.py"

_CONTENT = '    return {"activation": "none"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
