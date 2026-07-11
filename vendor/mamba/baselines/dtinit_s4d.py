import math

import torch


def init_delta(block):
    with torch.no_grad():
        dt = torch.exp(
            torch.rand(block.d_inner) * (math.log(1e-1) - math.log(1e-3))
            + math.log(1e-3)
        ).clamp(min=1e-4)
        inverse = dt + torch.log(-torch.expm1(-dt))
        block.dt_proj.bias.copy_(inverse)
        block.dt_const.copy_(inverse)
