"""RainNet-augmented baseline for image-harmonization (region-aware normalization).

The mask-conditioned U-Net PLUS the RainNet region-aware normalization (RAIN) modules
(Ling et al., "Region-aware Adaptive Instance Normalization for Image Harmonization",
CVPR 2021), which additionally transfer the BACKGROUND feature statistics onto the
FOREGROUND features. Provided for completeness. NOTE: at this reduced synthetic scale
(a global per-channel affine composite that the mask-conditioned U-Net already inverts
well) the extra RAIN transfer does NOT beat the plain mask-conditioned net -- the RAIN
advantage in the real iHarmony4 benchmark comes from complex, spatially-varying,
content-dependent appearance gaps. The task therefore uses copy/blind/mask as its
ordered baselines; see the onboarding notes.
"""


def get_network_config():
    return {"arch": "rain"}
