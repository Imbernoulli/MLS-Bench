"""Agent-editable optimizer for a fixed epoch-based LR schedule."""
from __future__ import annotations


# EDITABLE REGION
def build_optimizer(params):
    import torch

    return torch.optim.Adam(params, lr=3.5e-4, weight_decay=5e-4)
# END EDITABLE REGION
