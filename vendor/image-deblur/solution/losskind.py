"""Agent-editable surface: the RECONSTRUCTION LOSS FUNCTION (L2 vs Charbonnier+edge).

A FIXED compact residual deblur net (global residual ON, single-scale, sharp target -- all
else fixed at the strong reference) restores sharp images from motion-blurred ones. You
design ONLY the reconstruction LOSS: plain L2 (MSE) vs a robust Charbonnier loss plus an
edge (image-gradient) term; everything else is fixed.

    def get_loss_config():
        # kind: 'l2' (MSE) or 'charbonnier' (robust L1-like sqrt(e^2+eps^2))
        # edge_weight: weight of an image-gradient (edge) term (0 disables)
        return {'kind': 'charbonnier', 'edge_weight': 0.1, 'target_smooth': 0.0}

L2 (MSE) penalises large errors quadratically, which over-smooths the restored image (the
conditional-mean blur), under-restoring high-frequency detail. A robust Charbonnier loss
(Lai et al. LapSRN; used by MPRNet, Zamir et al. CVPR 2021) plus an edge/gradient term is
far less prone to over-smoothing and restores sharper edges -> higher deblur PSNR.

The DEFAULT returns plain L2 (edge_weight 0) -> over-smoothed, lower deblur PSNR. Charbonnier
+ edge gives clear PSNR headroom. A malformed / crashing return falls back to Charbonnier+edge."""
from __future__ import annotations







# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_loss_config():
    # Default: plain L2 (MSE), no edge term -> over-smoothed, lower deblur PSNR.
    return {"kind": "l2", "edge_weight": 0.0, "target_smooth": 0.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
