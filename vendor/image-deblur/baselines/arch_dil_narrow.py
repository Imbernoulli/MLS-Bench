"""WEAK dilation baseline: dilation=1 (narrow receptive field) -> cannot cover large blur."""
def get_arch_config():
    return {"dilation": 1}
