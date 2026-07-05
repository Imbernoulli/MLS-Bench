"""Weak baseline for cv-count-multiscale: SINGLE-scale context (identity).

No multi-scale context aggregation: the density tail sees only the single-scale
frontend features. Objects at scales the frontend's receptive field does not match are
mis-counted -> higher counting MAE.
"""


def build_context(cin):
    import torch.nn as nn
    return nn.Identity()
