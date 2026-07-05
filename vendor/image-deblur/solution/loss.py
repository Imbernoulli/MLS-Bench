"""Agent-editable surface: the RECONSTRUCTION-TARGET / LOSS for the deblur net.

A FIXED compact residual deblur net (global residual ON, single-scale fixed) is trained
to restore sharp images from motion-blurred ones. You design ONLY what the network is
optimised toward -- the reconstruction loss and, crucially, the TARGET it is matched to.

    def get_loss_config():
        # kind:          'l2' (MSE) or 'charbonnier' (robust L1-like sqrt(e^2+eps^2))
        # edge_weight:   weight of an extra image-GRADIENT (edge) term (0 disables)
        # target_smooth: sigma of a Gaussian LOW-PASS applied to the sharp GT BEFORE the
        #                loss. >0 optimises toward an OVER-SMOOTHED target (throws away the
        #                high-frequency detail the net should restore -> the classic
        #                L2-conditional-mean failure, LOW deblur PSNR). 0 = the TRUE SHARP
        #                target (correct -> HIGH deblur PSNR).
        return {'kind': 'charbonnier', 'edge_weight': 0.1, 'target_smooth': 0.0}

The single most important choice here is `target_smooth`: a deblur network can only be as
sharp as the target it is trained to match. If you optimise it toward a blurred (low-pass)
version of the ground truth, it learns to REPRODUCE the blur -- deblur PSNR collapses,
often below the do-nothing blurry-input floor. Optimising toward the true sharp GT (plus a
robust Charbonnier loss and an edge term that rewards high-frequency detail, as in
LapSRN / MPRNet) restores genuinely sharp images and maximises deblur PSNR.

The DEFAULT below optimises toward an OVER-SMOOTHED target (`target_smooth=1.2`) -> the
naive over-smoothing baseline (low deblur PSNR). Setting `target_smooth=0` (the true sharp
target) gives large PSNR headroom. A malformed / crashing return falls back to the sharp
target (`target_smooth=0.0`).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the reconstruction target / loss below
# ================================================================
def get_loss_config():
    # Default: optimise toward an OVER-SMOOTHED target -> reproduces blur, low deblur PSNR.
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 1.2}
# ================================================================
# END EDITABLE REGION
# ================================================================
