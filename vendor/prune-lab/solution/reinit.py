"""Agent-editable surface for prune-reinit (REAL CIFAR-10 / ResNet-18 pruning).

Choose the post-prune weight state for the survivors: keep the trained values, randomly reinitialize, or rewind to the checkpoint-provided early state.

Contract:
def reinit() -> str  # one of {"keep", "rewind", "random"}

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def reinit():
    # Native random-reinitialization choice.
    return "random"
