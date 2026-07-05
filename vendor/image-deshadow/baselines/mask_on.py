"""STRONG mask baseline: mask-guided U-Net -- the soft shadow mask is concatenated as a 4th
input channel (SP+M-Net, Le et al. ICCV 2019)."""


def get_mask_config():
    return {"use_mask": True}
