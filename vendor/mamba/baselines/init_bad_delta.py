"""Bad-Delta init — correct A, but Delta initialized far too large (all-forget)."""


def init_ssm(block):
    import torch
    from einops import repeat
    with torch.no_grad():
        A = repeat(torch.arange(1, block.d_state + 1, dtype=torch.float32),
                   "n -> d n", d=block.d_inner).contiguous()
        block.A_log.copy_(torch.log(A))
        big = torch.full((block.d_inner,), 3.0)   # softplus(3) ~ 3.05 >> 0.1
        block.dt_proj.bias.copy_(big)
        block.dt_const.copy_(big)
