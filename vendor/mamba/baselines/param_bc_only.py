"""Partial-selective parameterization — input-dependent B, C but CONSTANT dt.

An intermediate: the read/write projections B, C are input-dependent (as in S6),
but the timestep dt is a fixed per-channel constant (as in LTI). This recovers
some selectivity but is weaker than full S6 (dt selectivity is what most sharply
controls the "remember-vs-forget" content gate). A middle reference between the
LTI baseline and full selective.
"""


def parameterize(block, x, b, l):
    import torch
    from einops import rearrange, repeat
    x_dbl = block.x_proj(rearrange(x, "b d l -> (b l) d"))
    _dt, B, C = torch.split(x_dbl, [block.dt_rank, block.d_state, block.d_state], dim=-1)
    B = rearrange(B, "(b l) n -> b n l", l=l).contiguous()
    C = rearrange(C, "(b l) n -> b n l", l=l).contiguous()
    # dt is input-INDEPENDENT (constant per channel)
    dt = repeat(torch.zeros_like(block.dt_const), "d -> b d l", b=b, l=l).contiguous()
    return dt, B, C, block.dt_const.float()
