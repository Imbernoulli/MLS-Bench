"""Baseline edit (act_identity) for the conv_act surface.
Reference: vendor/mamba/baselines/act_identity.py
"""

_FILE = "mamba/solution/conv_act.py"

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
