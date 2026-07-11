"""Baseline edit (dt_identity) for the finalize_dt surface.
Reference: vendor/mamba/baselines/dt_identity.py
"""

_FILE = "mamba/solution/delta_softplus.py"

_CONTENT = '    return {"activation": "identity"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
