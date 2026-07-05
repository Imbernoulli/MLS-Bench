"""Agent-editable surface: CHANNEL WIDTH of the deblur backbone.

A FIXED encoder-decoder deblur net (global residual ON, sharp-target loss, moderate depth,
channel attention on -- all else fixed at the strong reference) restores sharp images from
motion-blurred ones. You design ONLY the base CHANNEL WIDTH of the backbone (the number of
feature channels; deeper stages scale x2/x4 from it); everything else is fixed.

    def get_arch_config():
        return {'width': 32}   # base channels (8 narrow .. 64 wide)

A too-NARROW backbone has too few feature channels to represent the spatially-varying deblur
correction, so it under-fits and leaves the output blurry (low deblur PSNR). Wider backbones
(more channels, as the width axis of EDSR / MPRNet) have the capacity to restore sharp detail
-> higher PSNR. Width is the complementary capacity axis to depth.

The DEFAULT returns width=12 (narrow) -> under-fits, lower PSNR. width=32 gives clear PSNR
headroom. A malformed / crashing return falls back to the strong reference width."""
from __future__ import annotations









# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_arch_config():
    # Default: a NARROW backbone (12 base channels) -> under-fits, lower deblur PSNR.
    return {"width": 12}
# ================================================================
# END EDITABLE REGION
# ================================================================
