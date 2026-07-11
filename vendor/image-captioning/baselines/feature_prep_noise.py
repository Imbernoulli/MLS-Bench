"""Design the CLIP-Feature Preprocessing — weak baseline (noise).

Reference implementation for the caption-feature-prep surface (preprocess_clip). See tasks/caption-feature-prep/edits/noise.edit.py.
"""
import torch


def preprocess_clip(emb):
    # Ship a noisy corruption of the cached CLIP features (deterministic seed).
    g = torch.Generator().manual_seed(0)
    return emb + 1.0 * emb.std() * torch.randn(emb.shape, generator=g, dtype=emb.dtype)
