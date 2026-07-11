"""Head baseline: identity pass-through (raw pooled features)."""
def build_head(feat_dim):
    import torch.nn as nn
    head = nn.Identity()
    head.name = "raw"
    return head
