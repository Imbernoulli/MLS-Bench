"""Weak baseline: plain SGD, NO momentum (slow, noisy convergence in few steps).
Reference: vendor/torchreid-reid/baselines/optim_sgd.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_optimizer(params):
    import torch

    return torch.optim.SGD(
        params, lr=3.5e-4, momentum=0.0, weight_decay=5e-4
    )
