"""Agent-editable surface: CHANNEL ATTENTION in the deblur backbone.

A FIXED encoder-decoder deblur net (global residual ON, sharp-target loss, resize-conv
upsampling, moderate depth -- all else fixed at the strong reference) restores sharp images
from motion-blurred ones. You design ONLY whether each ResBlock carries a CHANNEL-ATTENTION
(squeeze-and-excitation) gate; everything else is fixed.

    def get_arch_config():
        return {'attention': True}   # add an SE / channel-attention block (MPRNet CAB)

Channel attention (SENet, Hu et al. 2018; the Channel-Attention Block of MPRNet, Zamir et
al. CVPR 2021) lets the net re-weight feature channels to emphasise the high-frequency
channels that carry the deblur correction -> sharper restorations, higher deblur PSNR.

The DEFAULT returns attention=False (plain ResBlocks) -> lower PSNR. attention=True gives
clear PSNR headroom. A malformed / crashing return falls back to the strong reference."""
from __future__ import annotations










# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_arch_config():
    # Default: NO channel attention (plain ResBlocks) -> lower deblur PSNR.
    return {"attention": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
