"""Mask-conditioned baseline for image-harmonization (the strong region-aware design).

The MASK-CONDITIONED U-Net (composite RGB + the foreground mask, 4-ch input): it knows
exactly which region is the pasted foreground and recolours only it while preserving the
background -- the mask-conditioning that every real harmonizer relies on (DoveNet, Cong et
al. CVPR 2020; RainNet, Ling et al. CVPR 2021) -> the highest foreground PSNR (strong
reference / SOTA design at this scale).
"""


def get_network_config():
    return {"arch": "mask"}
