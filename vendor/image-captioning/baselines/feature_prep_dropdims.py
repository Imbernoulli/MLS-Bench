"""Reference implementation for the caption-feature-prep surface (preprocess_clip).

WEAK baseline candidate (dropdims): zero out a FIXED random 50% of the CLIP
feature dimensions before the mapping — measured on the k1 anchor alongside
raw/l2norm; see tasks/caption-feature-prep/leaderboard.csv.
"""

import torch


def preprocess_clip(emb):
    g = torch.Generator().manual_seed(0)
    keep = (torch.rand(emb.shape[-1], generator=g) > 0.5).to(emb.dtype)
    return emb * keep
