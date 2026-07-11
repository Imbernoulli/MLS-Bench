"""Baseline edit (dt_relu) for the finalize_dt surface.
Reference: vendor/mamba/baselines/dt_relu.py
"""

_FILE = "mamba/solution/delta_softplus.py"

_CONTENT = '    return {"activation": "relu"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
