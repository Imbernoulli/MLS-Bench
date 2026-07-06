"""Do-nothing floor baseline for image-harmonization (the identity).

The input-copy identity: return the composite unchanged (NO harmonization). Scores
exactly the composite-input foreground PSNR (fg_psnr == comp_fg_psnr, gain 0) -- the
floor any real harmonizer must beat.
"""


def get_network_config():
    return {"arch": "copy"}
