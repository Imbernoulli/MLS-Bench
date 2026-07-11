"""Adam optimizer baseline under the harness-owned fixed LR schedule.
Reference: vendor/torchreid-reid/baselines/optim_adam.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_optimizer(params):
    import torch

    return torch.optim.Adam(params, lr=3.5e-4, weight_decay=5e-4)
