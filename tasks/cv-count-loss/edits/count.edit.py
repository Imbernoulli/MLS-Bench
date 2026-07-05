"""Good baseline for cv-count-loss: pixel MSE + COUNT-CONSISTENCY. Directly supervises the integrated count -> lower MAE. Ref: vendor/crowd-counting/baselines/loss_count.py"""

_FILE = "crowd-counting/solution/loss.py"

_CONTENT = '    import torch\n    import torch.nn.functional as F\n    # foreground-weighted pixel loss: weight = 1 + alpha * (gt > 0)\n    fg = (gt > 1e-6).float()\n    w = 1.0 + 9.0 * fg\n    px = (w * (pred - gt) ** 2).mean()\n    # explicit count-consistency on the integrated mass\n    pc = pred.sum(dim=(-2, -1)); gc = gt.sum(dim=(-2, -1))\n    count_term = (pc - gc).abs().mean()\n    return px + 0.02 * count_term'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 34, "content": _CONTENT},
]
