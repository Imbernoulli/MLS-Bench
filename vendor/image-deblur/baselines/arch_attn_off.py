"""WEAK attention baseline: plain ResBlocks (no channel attention) -> lower deblur PSNR."""
def get_arch_config():
    return {"attention": False}
