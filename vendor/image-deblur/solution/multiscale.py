"""Agent-editable surface: SINGLE-SCALE vs MULTI-SCALE coarse-to-fine deblurring.

A FIXED compact residual deblur net (global residual ON, loss fixed) with SHARED
weights across scales is trained to restore sharp images from motion-blurred ones.
You design ONLY the number of pyramid scales in the coarse-to-fine recurrence.

    def get_scale_config():
        # scales: 1 (single-scale, full-res only) .. 3 (coarse-to-fine pyramid)
        return {'scales': 3}

With scales=1 the net deblurs at full resolution in one pass. With scales>1 the SAME
shared-weight net runs coarse-to-fine as a SCALE-RECURRENT network (SRN, Tao et al., CVPR
2018; cf. DeepDeblur, Nah et al., CVPR 2017): it first deblurs a downsampled (coarse)
image where the same motion blur spans fewer pixels and is easier to invert, then, at each
finer scale, feeds the net BOTH the pristine blurry image at that resolution (so full-res
detail is never lost) AND the upsampled coarser deblurred estimate as guidance, plus a
final full-resolution recurrence. Large motion blur is much easier to remove coarse-to-
fine, so a 3-scale pyramid gives sharper restorations and higher deblur PSNR than a single
full-res pass.

The DEFAULT below returns scales=1 (single-scale) -> the harder full-res-only baseline.
Using scales=3 gives clear PSNR headroom. A malformed / crashing return falls back to
scales=3.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the scale configuration below
# ================================================================
def get_scale_config():
    # Default: single-scale (full-res only) -> harder on large blur, lower PSNR.
    return {"scales": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
