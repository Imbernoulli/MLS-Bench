"""STRONG dilation baseline: dilated bottleneck convs (rates 2,4) -> large receptive field
(ASPP/multi-context-deshadow style), covers big soft shadows -> higher shadow-region PSNR.
"""


def get_dilation_config():
    return {"dilations": [2, 4]}
