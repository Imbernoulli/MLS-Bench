"""Agent-editable surface for prune-recovery-distill (REAL CIFAR-10 / ResNet-18 pruning).

Design the recovery objective, with access to the fixed dense teacher logits, for a pruned ResNet-18 under the fixed CIFAR-10 budget.

Contract:
def recovery_loss(logits, targets, teacher_logits) -> scalar Tensor

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def recovery_loss(logits, targets, teacher_logits):
    # Native cross-entropy implementation.
    return F.cross_entropy(logits, targets)
