"""residual STRONG (pre-norm residual): h + block_out (identity skip)."""
def residual_step(h, block_out):
    return h + block_out
