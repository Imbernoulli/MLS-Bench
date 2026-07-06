"""Design surface: HOW the foreground MASK is fed to the harmonizer.

Image harmonization must recolour the PASTED FOREGROUND while preserving the
already-correct BACKGROUND. Whether (and how) the network is told which region is the
foreground is the core inductive bias. Edit ONLY get_mask_conditioning() below to return
one of:

    'none'  -> MASK-BLIND: the net sees only the composite RGB. It is region-agnostic and
               cannot tell the foreground from the background, so it applies a compromise
               correction that disturbs the already-correct background and only partially
               fixes the foreground (weak).
    'concat'-> the foreground mask is CONCATENATED as a 4th input channel (DoveNet, Cong et
               al. CVPR 2020): the net knows exactly which region to recolour.
    'gated' -> concat PLUS a mask-gated output blend that hard-restricts the edit to the
               foreground, so the (already-correct) background is provably preserved and all
               capacity goes to the foreground -> the strongest region-conditioning.

Everything else (data, width/depth, normalization, loss, optimiser, iters, seed, eval
split, the foreground-region PSNR metric) is FIXED. A malformed / crashing return falls
back to 'concat'.
"""


def get_mask_conditioning():
    # Default: MASK-BLIND (region-agnostic) -> partial recovery only.
    return "none"
