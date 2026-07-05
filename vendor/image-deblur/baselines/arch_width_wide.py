"""STRONG width baseline: wide backbone (32 base channels) -> more capacity, higher PSNR.
Width axis of restoration nets: EDSR (Lim et al. CVPR 2017), MPRNet (Zamir et al. CVPR 2021)."""
def get_arch_config():
    return {"width": 32}
