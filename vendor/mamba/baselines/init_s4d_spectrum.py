"""Initialize A to the real S4D spectrum -(1, ..., N)."""

import torch
from einops import repeat


def init_state(block):
    with torch.no_grad():
        value = repeat(
            torch.arange(1, block.d_state + 1, dtype=torch.float32),
            "n -> d n", d=block.d_inner,
        ).contiguous()
        block.A_log.copy_(torch.log(value))
