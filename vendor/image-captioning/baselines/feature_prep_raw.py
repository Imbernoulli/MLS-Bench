"""Design the CLIP-Feature Preprocessing — strong baseline (raw).

Reference implementation for the caption-feature-prep surface (preprocess_clip). See tasks/caption-feature-prep/edits/raw.edit.py.
"""
import torch


def preprocess_clip(emb):
    # Identity: pass the raw cached CLIP embedding straight to the mapping.
    return emb
