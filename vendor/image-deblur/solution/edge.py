"""Agent-editable surface: the EDGE / GRADIENT LOSS WEIGHT.

A FIXED compact residual deblur net (global residual ON, single-scale, Charbonnier
reconstruction loss on the true SHARP target -- all else fixed at the strong reference)
restores sharp images from motion-blurred ones. You design ONLY the weight of an extra EDGE
(image-gradient) loss term; everything else is fixed.

    def get_loss_config():
        return {'kind': 'charbonnier', 'edge_weight': 0.5, 'target_smooth': 0.0}

Motion blur destroys HIGH-FREQUENCY detail (edges); a plain reconstruction loss (L1 /
Charbonnier) is dominated by the low-frequency bulk and under-penalises residual edge blur,
so the output stays slightly soft. Adding an edge / gradient loss that matches the image
GRADIENTS of the restored and sharp images (LapSRN, Lai et al. CVPR 2017; MPRNet edge loss)
explicitly rewards restoring sharp edges -> higher deblur PSNR.

The DEFAULT returns edge_weight=0.0 (no edge term) -> edges under-restored, lower PSNR.
edge_weight=0.5 gives clear PSNR headroom. A malformed return falls back to the strong ref."""
from __future__ import annotations








# ================================================================
# EDITABLE REGION — design the configuration below
# ================================================================
def get_loss_config():
    # Default: NO edge term (edge_weight=0.0) -> edges under-restored, lower deblur PSNR.
    return {"kind": "charbonnier", "edge_weight": 0.0, "target_smooth": 0.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
