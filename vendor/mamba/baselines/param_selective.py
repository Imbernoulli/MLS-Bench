"""Selective (S6 / Mamba) parameterization — the strong reference.

dt, B, C are all computed as FUNCTIONS OF THE INPUT x via x_proj / dt_proj, so the
recurrence content-gates which tokens enter the state. This is the real Mamba
selective-scan wiring; it solves selective copying near-perfectly.
"""


def parameterize(block, x, b, l):
    import torch
    from einops import rearrange
    x_dbl = block.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt, B, C = torch.split(x_dbl, [block.dt_rank, block.d_state, block.d_state], dim=-1)
    dt = block.dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=l)
    B = rearrange(B, "(b l) n -> b n l", l=l).contiguous()
    C = rearrange(C, "(b l) n -> b n l", l=l).contiguous()
    return dt, B, C, block.dt_proj.bias.float()
