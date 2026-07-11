"""Design the Per-Token Loss Weighting — strong baseline (uniform).

Reference implementation for the caption-token-weighting surface (token_weights). See tasks/caption-token-weighting/edits/uniform.edit.py.
"""
import torch


def token_weights(targets, pad_id, ctx):
    # Uniform token weights: every (non-pad) caption token contributes equally.
    return torch.ones_like(targets, dtype=torch.float)
