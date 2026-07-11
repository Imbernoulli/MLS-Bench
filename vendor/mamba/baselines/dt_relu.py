"""delta MID: ReLU(dt+bias) -> non-negative but with a hard zero region (many dead
Delta=0 channels -> no state update there) -> weaker than smooth softplus."""
def finalize_dt(block, dt):
    import torch
    from einops import rearrange
    return torch.relu(dt + rearrange(block.dt_proj.bias.float(), "d -> 1 d 1"))
