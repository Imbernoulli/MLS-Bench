"""Agent-editable surface: PHYSICS PARAMETERISATION of the deshadower output.

A FIXED mask-guided deshadower (same U-Net width/depth, loss, optimiser, data, iters, seed,
eval split) removes a cast shadow. Under the SP+M-Net physics (Le et al. ICCV 2019) the clean
image is a per-pixel AFFINE relighting of the shadowed one:  J = w * I + b. You design ONLY
how the net produces its output:

    def get_physics_config():
        # {'mode': 'residual' | 'physics'}
        return {'mode': 'physics'}

  * mode='residual' -> the net predicts a free 3-ch RESIDUAL added to the shadowed input
                       (clean = shadowed + net(.)). Unconstrained: it can drift the lit
                       region and does not respect the multiplicative illumination structure.
  * mode='physics'  -> the SP+M-Net ILLUMINATION MODEL: the net predicts per-pixel affine
                       relighting parameters (w, b) and outputs J = w*I + b, a VALID
                       multiplicative-illumination inverse initialised near identity. It
                       matches the true degradation form -> higher shadow-region PSNR.

The DEFAULT below returns mode='residual' (the unconstrained free residual). The physics
parameterisation matches the degradation and typically raises shadow-region PSNR. A malformed
/ crashing return falls back to mode='residual' (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the physics parameterisation below
# ================================================================
def get_physics_config():
    # Default: free unconstrained residual (no physics parameterisation).
    return {"mode": "residual"}
# ================================================================
# END EDITABLE REGION
# ================================================================
