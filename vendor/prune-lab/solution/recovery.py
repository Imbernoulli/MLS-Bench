"""Agent-editable surface for prune-recovery (REAL CIFAR-10 / ResNet-18 pruning).

Design the RECOVERY procedure that re-adapts a magnitude-pruned ResNet-18 back to accuracy within a fixed fine-tune budget on real CIFAR-10.

Contract:
def recover(model, masked_finetune, cfg):
# call masked_finetune(epochs=.., lr=..) to train with the mask re-applied
# each step; cfg has keys epochs/lr/batch. Calls must consume cfg["epochs"] exactly.

The native implementation below is evaluated directly through this surface.
A malformed, crashing, wrong-shape, or non-finite return invalidates verification.
"""
from __future__ import annotations
import torch

def recover(model, masked_finetune, cfg):
    # Native full-budget recovery with a conservative learning rate.
    masked_finetune(epochs=cfg["epochs"], lr=min(float(cfg["lr"]), 0.0025))
