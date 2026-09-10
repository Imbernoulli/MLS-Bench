"""Baseline: no regularization (lower bound).

Only the photometric loss (0.8 * L1 + 0.2 * SSIM) supervises the
optimization; `compute_regularizer` returns zero. Any meaningful
regularizer should beat this baseline.
"""

_FILE = "gsplat/custom_regularizer.py"

_NONE = '''
def compute_regularizer(splats, step, scene_scale):
    """No regularization — zero added to the photometric loss."""
    return torch.zeros((), device=splats["means"].device)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 37,
        "end_line": 51,
        "content": _NONE,
    },
]
