"""bc-coupling DEGENERATE: read C := write B (tied read/write direction)."""
def couple_bc(block, B, C_lowrank, b, l):
    return B.contiguous()
