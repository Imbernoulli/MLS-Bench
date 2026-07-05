"""WEAK residual baseline: predict the FULL image directly (no global residual).
Harder optimisation at this budget -> blurrier output, lower deblur PSNR."""
def get_residual_config():
    return {"global_residual": False}
