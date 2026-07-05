"""Agent-editable surface: DILATION / RECEPTIVE FIELD of the bottleneck.

A FIXED encoder-decoder deblur net (global residual ON, sharp-target loss, resize-conv
upsampling, channel attention on, moderate depth -- all else fixed at the strong reference)
restores sharp images from motion-blurred ones. You design ONLY the DILATION of the
bottleneck convolutions, which sets the RECEPTIVE FIELD; everything else is fixed.

    def get_arch_config():
        return {'dilation': 4}   # wide receptive field (1 narrow .. 4 wide)

A motion-blur kernel spreads each sharp pixel across a STREAK; to invert it the net must SEE
the whole streak. If the receptive field is smaller than the blur, the net cannot gather the
information to deblur and the output stays blurry (low PSNR). Dilated convolutions widen the
receptive field WITHOUT extra parameters (Yu & Koltun 2016) -> covers a larger streak, higher
PSNR, in the heavy-blur band.

The DEFAULT returns dilation=1 (narrow RF) -> cannot cover large blur, lower PSNR. dilation=4
gives clear PSNR headroom. A malformed / crashing return falls back to the strong ref."""
from __future__ import annotations








# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_arch_config():
    # Default: dilation=1 (narrow receptive field) -> cannot cover large blur, lower PSNR.
    return {"dilation": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
