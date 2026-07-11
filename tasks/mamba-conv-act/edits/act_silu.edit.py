"""Baseline edit (act_silu) for the conv_act surface.
Reference: vendor/mamba/baselines/act_silu.py
"""

_FILE = "mamba/solution/conv_act.py"

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
