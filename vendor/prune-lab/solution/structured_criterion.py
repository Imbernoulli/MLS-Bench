"""Agent-editable surface for prune-structured-criterion (REAL CIFAR-10 / ResNet-18 pruning).

Choose channel importance for dependency-aware structured ResNet-18 pruning at the harness-requested ratio, with the realized MAC reduction measured on CIFAR-10.

Contract:
def importance_spec() -> dict  # e.g. {"type": "l1"} | {"type": "l2"}
# | {"type": "taylor", "batches": 8} | {"type": "bn"} | {"type": "random"}

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def importance_spec():
    # Native random channel-importance implementation.
    return {"type": "random"}
