"""Weak baseline for cv-count-upsample: NO decoder (coarse stride-8). Objects in one cell can't be separated -> higher MAE. Ref: vendor/crowd-counting/baselines/upsample_none.py"""

_FILE = "crowd-counting/solution/upsample.py"

_CONTENT = '    import torch.nn as nn\n    return nn.Identity()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 38, "content": _CONTENT},
]
