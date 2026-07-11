"""Design the Input-Feature Regularization — strong baseline (gauss).

Reference implementation for the caption-feature-augment surface (augment_clip). See tasks/caption-feature-augment/edits/gauss.edit.py.
"""
import torch


def augment_clip(emb, training):
    # Train-time regularisation only: small Gaussian jitter + light feature
    # dropout curb over-fitting to the cached CLIP vectors. Eval is unchanged.
    if not training:
        return emb
    x = emb + 0.01 * torch.randn_like(emb)
    keep = (torch.rand_like(x) > 0.1).float()
    return x * keep
