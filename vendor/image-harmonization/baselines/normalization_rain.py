"""image-harmonization normalization baseline: rain.

RainNet region-aware AdaIN (Ling et al. CVPR 2021): transfers BG stats onto FG (SOTA).
"""


def get_normalization():
    return 'rain'
