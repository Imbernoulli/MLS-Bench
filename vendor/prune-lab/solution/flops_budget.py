"""Agent-editable surface for prune-flops-budget (REAL CIFAR-10 / ResNet-18 pruning).

Choose channel importance for structured ResNet-18 pruning under the harness-enforced measured-MAC budget on CIFAR-10.

Contract:
def importance_spec() -> dict  # channel importance used while meeting the FLOPs budget

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def importance_spec():
    # Native random channel-importance implementation.
    return {"type": "random"}
