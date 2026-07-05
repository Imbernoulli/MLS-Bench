"""STRONG depth baseline: deeper stack (3 ResBlocks/stage) -> more capacity, higher PSNR.
Deep ResBlock stacks: DeepDeblur (Nah et al. CVPR 2017), MPRNet (Zamir et al. CVPR 2021)."""
def get_arch_config():
    return {"n_resblocks": 3}
