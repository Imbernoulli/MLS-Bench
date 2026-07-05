"""Agent-editable surface: the DENSITY-MAP TRAINING LOSS.

Define `density_loss(pred, gt)` -> a scalar loss tensor, where `pred` and `gt` are
`(B, h, w)` density maps on the stride-8 grid (gt integrates to DENSITY_SCALE * count).
Everything else (data, backbone, density head, optimiser, iterations) is FIXED; only the
loss changes, so any change in counting MAE is attributable to the loss design.

The counting metric measures the INTEGRATED COUNT, but a plain pixel-wise MSE only
supervises per-pixel density and constrains the total mass weakly -> per-image counts
drift. Adding an explicit COUNT-CONSISTENCY term (penalise |sum(pred) - sum(gt)|)
directly supervises what the metric measures and lowers counting MAE. This mirrors
DM-Count-style explicit counting supervision (Wang et al., NeurIPS 2020) and count-aware
losses layered on pixel MSE.

    def density_loss(pred, gt):
        import torch.nn.functional as F
        mse = F.mse_loss(pred, gt)
        count_term = (pred.sum(dim=(-2,-1)) - gt.sum(dim=(-2,-1))).abs().mean()
        return mse + 0.05 * count_term

The DEFAULT below is the deliberately weak PLAIN pixel MSE (no count term). A crashing
/ malformed loss falls back to the default per-step.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the density-map training loss below
# ================================================================
def density_loss(pred, gt):
    # Default: plain pixel-wise MSE only (weak). No explicit count supervision -> the
    # integrated count drifts -> higher counting MAE.
    import torch.nn.functional as F
    return F.mse_loss(pred, gt)
# ================================================================
# END EDITABLE REGION
# ================================================================
