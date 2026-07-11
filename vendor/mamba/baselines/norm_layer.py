"""norm STRONG (Transformer++/Mamba): LayerNorm over the feature dimension."""
def make_norm(d_model):
    import torch.nn as nn
    return nn.LayerNorm(d_model)
