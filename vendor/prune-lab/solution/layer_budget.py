"""Agent-editable surface for prune-layer-budget (REAL CIFAR-10 / ResNet-18 pruning).

Allocate a fixed global sparsity budget across ResNet-18 layers and measure the resulting CIFAR-10 test accuracy.

Contract:
def layer_sparsity(layer_names: list[str]) -> dict[str, float]
# per-layer sparsity in [0,1); missing layers default to the global target.

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def layer_sparsity(layer_names):
    # Native uniform allocation: missing entries use the global target.
    return {}
