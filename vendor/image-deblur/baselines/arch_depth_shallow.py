"""WEAK depth baseline: shallow net (1 ResBlock/stage) -> under-fits heavy blur, lower PSNR."""
def get_arch_config():
    return {"n_resblocks": 1}
