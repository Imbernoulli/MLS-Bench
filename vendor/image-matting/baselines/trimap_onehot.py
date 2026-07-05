"""Good baseline for cv-matting-trimap-encoding: 3-plane one-hot trimap.

Encodes the trimap as three planes (definite-fg / unknown / definite-bg), telling the
net exactly where the solved regions are so it only interpolates the soft matte across
the known boundary (Deep Image Matting, Xu et al. 2017 feeds the trimap as an extra
input) -> much lower SAD. Reference: vendor/image-matting/baselines/trimap_onehot.py
"""


def encode_trimap(trimap):
    import torch
    fg = (trimap > 0.75).float()
    bg = (trimap < 0.25).float()
    unk = ((trimap >= 0.25) & (trimap <= 0.75)).float()
    return torch.stack([fg, unk, bg], dim=1)
