"""GOOD residual baseline: predict a GLOBAL RESIDUAL (sharp = blurry + net(blurry)).
DeepDeblur / SRN / MPRNet all use this long identity skip."""
def get_residual_config():
    return {"global_residual": True}
