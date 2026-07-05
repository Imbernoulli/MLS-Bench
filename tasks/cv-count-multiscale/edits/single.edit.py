"""Weak baseline for cv-count-multiscale: SINGLE-scale context. Mis-counts off-scale objects -> higher MAE. Ref: vendor/crowd-counting/baselines/multiscale_single.py"""

_FILE = "crowd-counting/solution/multiscale.py"

_CONTENT = '    import torch.nn as nn\n    return nn.Identity()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 43, "content": _CONTENT},
]
