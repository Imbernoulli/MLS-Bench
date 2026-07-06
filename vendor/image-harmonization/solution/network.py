"""Design surface for the image-harmonization NETWORK task.

The design lever is HOW MUCH REGION INFORMATION the harmonizer uses -- the core
inductive bias of image harmonization is that the recolour must be applied to the
PASTED FOREGROUND while the already-correct BACKGROUND is preserved. Edit ONLY
get_network_config() below to choose the harmonizer architecture. Return a dict with an
'arch' key:

    'copy'  -> the input-copy identity (NO harmonization; the do-nothing floor).
    'blind' -> a MASK-BLIND encoder-decoder U-Net (composite RGB only): region-AGNOSTIC,
               it cannot tell the pasted foreground from the background, so it applies a
               compromise correction and only partially fixes the mismatch.
    'mask'  -> the MASK-CONDITIONED U-Net (composite RGB + the foreground mask): it knows
               exactly which region is the pasted foreground and recolours only it while
               preserving the background -- the region-aware design every real harmonizer
               relies on (DoveNet, Cong et al. CVPR 2020; RainNet, Ling et al. CVPR 2021).
    'rain'  -> the mask-conditioned U-Net PLUS the RainNet RAIN region-aware normalization
               modules (Ling et al. CVPR 2021).

Everything else (data, base width/depth, optimiser, iterations, seed, eval split, the
synthetic composite degradation, the foreground-region PSNR metric) is FIXED. A
malformed / crashing return falls back to arch='mask'.
"""


def get_network_config():
    # Default: MASK-BLIND region-AGNOSTIC U-Net (corrects the mismatch only partially).
    return {"arch": "blind"}
