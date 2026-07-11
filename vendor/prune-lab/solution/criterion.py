"""Agent-editable surface for prune-criterion (REAL CIFAR-10 / ResNet-18 pruning).

Design the per-weight IMPORTANCE CRITERION that decides which weights a trained ResNet-18 keeps at a fixed high sparsity on real CIFAR-10.

Contract:
def importance(name, weight, grad) -> Tensor  # same shape as weight,
# non-negative, finite; larger = more likely to be KEPT.

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def importance(name, weight, grad):
    # Native random-importance implementation.
    return torch.rand_like(weight)
