"""Agent-editable surface: the ALPHA-MATTE TRAINING LOSS.

Return a callable
    loss_fn(pred, gt, image, fg, bg, trimap, unknown) -> scalar torch tensor
where
    pred     (B,H,W) predicted alpha in [0,1]
    gt       (B,H,W) ground-truth alpha in [0,1]
    image    (B,3,H,W) the observed composite I
    fg, bg   (B,3,H,W) the foreground / background layers (I = a*fg + (1-a)*bg)
    trimap   (B,H,W) in {0,0.5,1}
    unknown  (B,H,W) bool: the trimap UNKNOWN band (where the matte is scored)

A fixed U-Net matting net is trained a few hundred steps with THIS loss, then scored
by SAD (sum of absolute alpha differences /1000) in the UNKNOWN band on a held-out
val split (LOWER is better).

    def get_matting_loss():
        def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
            u = unknown.float()
            d = u.sum(dim=(-2,-1)).clamp(min=1.0)
            alpha_l = ((pred - gt).abs() * u).sum(dim=(-2,-1)) / d          # alpha L1
            comp = pred.unsqueeze(1)*fg + (1-pred.unsqueeze(1))*bg
            comp_l = ((comp - image).abs().mean(1) * u).sum(dim=(-2,-1)) / d # composition L1
            return (alpha_l + 0.5*comp_l).mean()
        return loss_fn

The DEFAULT below is a deliberately weak UNIFORM WHOLE-IMAGE ALPHA-L1 loss: it
averages |pred - gt| over EVERY pixel (fg + bg + unknown). The trivial solid regions
(alpha exactly 0 or 1, which the net nails immediately and which dominate the image
area) swamp the mean, so the optimiser gets little gradient signal on the hard
UNKNOWN transition band -> the matte there is under-fit and SAD (measured in the
unknown band) stays high. The matting-standard fix is to (i) restrict the loss to
the UNKNOWN band (where the matte is actually solved) and (ii) add a COMPOSITION loss
(Deep Image Matting, Xu et al. 2017 — penalise |I - (a*F + (1-a)*B)|, tying the matte
to the observed image), optionally a Laplacian-pyramid / gradient term for sharper
edges. That focuses capacity on the transition and lowers SAD with clear headroom. A
malformed / crashing / non-finite loss falls back to the whole-image alpha-L1.
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the alpha-matte training loss below
# ================================================================
def get_matting_loss():
    # Default: UNIFORM whole-image alpha-L1 (averaged over EVERY pixel, not the
    # unknown band). The trivial solid fg/bg regions (alpha exactly 0 or 1, which
    # the net nails immediately) dominate the mean, so the gradient signal on the
    # hard UNKNOWN transition is diluted and the matte there is under-fit -> higher
    # SAD in the unknown band. Restricting the loss to the unknown band and adding a
    # composition term (Deep Image Matting) focuses capacity on the transition and
    # lowers SAD with clear headroom.
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        return (pred - gt).abs().mean()
    return loss_fn
# ================================================================
# END EDITABLE REGION
# ================================================================
