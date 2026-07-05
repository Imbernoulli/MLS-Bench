"""Agent-editable surface: RESIDUAL-BLOCK DEPTH of the deblur backbone.

A FIXED encoder-decoder deblur net (global residual ON, sharp-target loss, resize-conv
upsampling, channel attention on -- all else fixed at the strong reference) restores sharp
images from motion-blurred ones. You design ONLY the number of ResBlocks per enc/dec stage
(the bottleneck uses twice as many); everything else is fixed.

    def get_arch_config():
        return {'n_resblocks': 3}   # deeper stack -> more capacity for heavy blur

Deblurring a LARGE motion streak undoes a wide, structured degradation; a too-SHALLOW net
under-fits it and leaves the output blurry (low deblur PSNR). Adding ResBlocks (the deep
stacks of DeepDeblur, Nah et al. CVPR 2017, and MPRNet, Zamir et al. CVPR 2021) gives the
capacity to restore high-frequency detail -> higher PSNR, in the heavy-blur band.

The DEFAULT returns n_resblocks=1 (shallow) -> under-fits, low PSNR. n_resblocks=3 gives
clear PSNR headroom. A malformed / crashing return falls back to the strong reference."""
from __future__ import annotations









# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_arch_config():
    # Default: a SHALLOW net (1 ResBlock per stage) -> under-fits heavy blur, low PSNR.
    return {"n_resblocks": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
