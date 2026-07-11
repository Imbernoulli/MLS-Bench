"""Design the Input-Feature Regularization — weak baseline (noaug).

Reference implementation for the caption-feature-augment surface (augment_clip). See tasks/caption-feature-augment/edits/noaug.edit.py.
"""
import torch


def augment_clip(emb, training):
    # No feature augmentation: feed the CLIP embedding unchanged.
    return emb
