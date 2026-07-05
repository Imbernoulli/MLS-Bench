"""Weak baseline for cv-count-attention: NO attention (identity).

The features feed straight into the density tail. Unannotated distractor CLUTTER in the
background is not suppressed, so the model spends density mass on non-objects and
mis-counts -> higher counting MAE. This is the no-attention control.
"""


def build_attention(cin):
    import torch.nn as nn
    return nn.Identity()
