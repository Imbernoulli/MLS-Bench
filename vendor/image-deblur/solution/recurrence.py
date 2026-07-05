"""Agent-editable surface: WITHIN-SCALE RECURRENCE DEPTH.

A FIXED compact residual deblur net (global residual ON, sharp-target loss -- all else fixed
at the strong reference) is run RECURRENTLY at full resolution with SHARED weights. You
design ONLY the number of refinement passes; everything else is fixed.

    def get_recurrence_config():
        return {'n_recurrence': 4}   # full-res refinement passes (1..4)

This is the scale-recurrent refinement of SRN (Tao et al., CVPR 2018) applied WITHOUT the
multi-scale pyramid: the SAME weights are applied several times, so each pass starts from a
better-deblurred estimate and removes more of the (large) blur -- at IDENTICAL parameter
count. A single pass under-deblurs a heavy streak; more passes converge to a sharp image ->
higher deblur PSNR, in the heavy-blur band.

The DEFAULT returns n_recurrence=1 (one pass) -> under-deblurs, lower PSNR. n_recurrence=4
gives clear PSNR headroom. A malformed / crashing return falls back to the strong ref."""
from __future__ import annotations









# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_recurrence_config():
    # Default: a single full-res pass -> under-deblurs heavy blur, lower deblur PSNR.
    return {"n_recurrence": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
