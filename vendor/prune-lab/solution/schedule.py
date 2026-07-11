"""Agent-editable surface for prune-schedule (REAL CIFAR-10 / ResNet-18 pruning).

Design a one-shot or gradual pruning schedule under the fixed final sparsity and recovery budget on CIFAR-10.

Contract:
def schedule(target_sparsity, total_steps) -> list[(sparsity, epochs)]
# monotone non-decreasing sparsity ending at target_sparsity; epochs sum == total_steps.

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def schedule(target_sparsity, total_steps):
    # Native one-rung schedule consuming the complete budget.
    return [(target_sparsity, total_steps)]
