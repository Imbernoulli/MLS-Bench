"""Weak baseline for cv-count-attention: NO attention. Clutter not suppressed -> higher MAE. Ref: vendor/crowd-counting/baselines/attention_none.py"""

_FILE = "crowd-counting/solution/attention.py"

_CONTENT = '    import torch.nn as nn\n    return nn.Identity()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
