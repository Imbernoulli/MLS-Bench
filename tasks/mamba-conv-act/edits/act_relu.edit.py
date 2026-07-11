"""Baseline edit (act_relu) for the conv_act surface.
Reference: vendor/mamba/baselines/act_relu.py
"""

_FILE = "mamba/solution/conv_act.py"

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
