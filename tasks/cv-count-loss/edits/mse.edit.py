"""Weak baseline for cv-count-loss: plain pixel MSE only. No count supervision -> count drifts -> higher MAE. Ref: vendor/crowd-counting/baselines/loss_mse.py"""

_FILE = "crowd-counting/solution/loss.py"

_CONTENT = '    import torch.nn.functional as F\n    return F.mse_loss(pred, gt)'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 34, "content": _CONTENT},
]
