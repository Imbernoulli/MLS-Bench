"""Mask-blind baseline for image-harmonization (region-agnostic weak learned baseline).

A MASK-BLIND encoder-decoder U-Net (composite RGB only, 3-ch input): a region-agnostic
image-to-image net that CANNOT tell the pasted foreground from the background, so it
applies a compromise correction that disturbs the already-correct background and only
partially fixes the foreground -> a middling foreground PSNR (above the do-nothing floor,
below the mask-conditioned net).
"""


def get_network_config():
    return {"arch": "blind"}
