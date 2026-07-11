"""delta STRONG (Mamba): Delta = softplus(dt + bias) -> strictly positive, stable."""
def finalize_dt(block, dt):
    import torch.nn.functional as F
    from einops import rearrange
    return F.softplus(dt + rearrange(block.dt_proj.bias.float(), "d -> 1 d 1"))
