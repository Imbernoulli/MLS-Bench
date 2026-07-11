"""norm DEGENERATE: Identity (no normalization)."""
def make_norm(d_model):
    import torch.nn as nn
    return nn.Identity()
