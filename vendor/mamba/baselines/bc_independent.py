"""bc-coupling STRONG (Mamba): C is its OWN input-dependent projection, decoupled
from B."""
def couple_bc(block, B, C_lowrank, b, l):
    from einops import rearrange
    return rearrange(C_lowrank, "(b l) n -> b n l", l=l).contiguous()
