"""SGD-with-momentum baseline under the harness-owned fixed LR schedule.
Reference: vendor/torchreid-reid/baselines/optim_sgd_mom.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_optimizer(params):
    import torch

    return torch.optim.SGD(
        params, lr=3.5e-4, momentum=0.9, weight_decay=5e-4
    )
