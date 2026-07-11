"""Time-invariant (S4 / LTI) parameterization — the naive baseline.

dt, B, C are all input-INDEPENDENT constants broadcast over positions. This is a
classic linear-time-invariant state-space recurrence: it cannot content-gate the
randomly-scattered data tokens, so it fails selective copying (low accuracy).
"""


def parameterize(block, x, b, l):
    import torch
    from einops import repeat
    dt = repeat(torch.zeros_like(block.dt_const), "d -> b d l", b=b, l=l).contiguous()
    B = repeat(block.B_const, "n -> b n l", b=b, l=l).contiguous()
    C = repeat(block.C_const, "n -> b n l", b=b, l=l).contiguous()
    return dt, B, C, block.dt_const.float()
