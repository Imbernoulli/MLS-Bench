"""bc-coupling MID: read C is a FIXED (input-independent) constant direction.
Decoupled from B but NOT content-dependent -> partial retrieval."""
def couple_bc(block, B, C_lowrank, b, l):
    from einops import repeat
    return repeat(block.C_const, "n -> b n l", b=b, l=l).contiguous()
