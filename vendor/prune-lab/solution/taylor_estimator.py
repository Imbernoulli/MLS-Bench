"""Agent-editable surface for prune-taylor-estimator (REAL CIFAR-10 / ResNet-18 pruning).

Design the data-aware IMPORTANCE ESTIMATOR using the complete supplied calibration pass to rank ResNet-18 weights for pruning at an enforced sparsity on CIFAR-10.

Contract:
def estimate_importance(model, batches, params) -> dict[name, Tensor]
# batches: list of (x, y) tensors; params: list of (name, Parameter).

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def estimate_importance(model, batches, params):
    # Native magnitude implementation.
    return {name: p.detach().abs() for name, p in params}
