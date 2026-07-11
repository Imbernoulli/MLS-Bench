import torch


def init_delta(block):
    with torch.no_grad():
        value = torch.full((block.d_inner,), -12.0)
        block.dt_proj.bias.copy_(value)
        block.dt_const.copy_(value)
