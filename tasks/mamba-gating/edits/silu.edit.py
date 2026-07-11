"""Baseline edit (silu) for the gate surface.
Reference: vendor/mamba/baselines/gate_silu.py
"""

_FILE = "mamba/solution/gating.py"

_CONTENT = '    return {"activation": "silu"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
