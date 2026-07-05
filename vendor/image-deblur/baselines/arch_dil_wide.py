"""STRONG dilation baseline: dilation=4 (wide receptive field) -> covers larger streak.
Dilated conv receptive field: Yu & Koltun (2016)."""
def get_arch_config():
    return {"dilation": 4}
