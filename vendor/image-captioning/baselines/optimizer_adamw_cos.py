"""Design the Optimizer and LR Schedule — strong baseline (adamw_cos).

Reference implementation for the caption-optimizer surface (make_optimizer). See tasks/caption-optimizer/edits/adamw_cos.edit.py.
"""
import torch


def make_optimizer(params):
    # AdamW with decoupled weight decay — the standard, stable choice for a
    # small mapping-network fine-tune.
    return torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)


def lr_scale(step, total):
    # Cosine decay from 1.0 -> 0.1 over the budget: high LR early to move fast,
    # annealed late to settle into a sharper minimum.
    import math
    if total <= 0:
        return 1.0
    c = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    return 0.1 + 0.9 * c
