"""Design the CLIP-Feature Preprocessing — strong baseline (l2norm).

Reference implementation for the caption-feature-prep surface (preprocess_clip). See tasks/caption-feature-prep/edits/l2norm.edit.py.
"""
import torch


def preprocess_clip(emb):
    # Row-wise L2-renormalise the CLIP features to unit length (the cosine
    # geometry CLIP was trained in), giving the mapping a stable input scale.
    x = emb.float()
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
