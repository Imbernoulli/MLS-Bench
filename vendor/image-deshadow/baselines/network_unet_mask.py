"""STRONG network baseline (the good answer = SOTA reference): the MASK-GUIDED U-Net. The
soft shadow mask is concatenated as a 4th input channel, so the net knows exactly WHERE and
HOW MUCH to brighten -- the SP+M-Net (Le et al. ICCV 2019) physically-parameterised recovery
that fits the multiplicative attenuation I = a*J. Highest shadow-region PSNR."""
def get_network_config():
    return {"arch": "unet_mask"}
