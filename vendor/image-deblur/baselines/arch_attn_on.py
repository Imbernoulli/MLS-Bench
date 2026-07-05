"""STRONG attention baseline: channel attention (SE/CAB) on -> sharper, higher PSNR.
Channel Attention Block: MPRNet (Zamir et al. CVPR 2021); SENet (Hu et al. 2018)."""
def get_arch_config():
    return {"attention": True}
