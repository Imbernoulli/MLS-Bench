"""Agent-editable surface for prune-second-order (REAL CIFAR-10 / ResNet-18 pruning).

Design a non-negative importance using the supplied ResNet-18 weights, gradients, and Fisher-diagonal proxy at the enforced CIFAR-10 sparsity.

Contract:
def importance2(name, weight, grad, fisher) -> Tensor  # fisher = E[grad^2] proxy (or None)

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def importance2(name, weight, grad, fisher):
    # Native magnitude implementation.
    return weight.abs()
