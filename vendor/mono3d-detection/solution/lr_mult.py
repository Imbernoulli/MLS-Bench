"""Monocular-3D LEARNING-RATE design surface (agent-editable) for mono3d-learning-rate.

The depth head learns a multiplicative residual on top of the analytic geometry depth Z=f*H/h2d.
Whether it actually LEARNS that residual within the fixed OneCycle schedule depends on its
learning rate.

You implement:

    def build_lr_mult() -> float:

Return the multiplier applied to the base learning rate for the DEPTH head's parameters (the
head under study). The base LR and schedule are fixed; only this multiplier changes.

The DEFAULT below is the WEAK baseline: a TINY multiplier (0.01), so the head barely trains — the
learned residual stays ~0 and the model reduces to raw analytic geometry, unable to correct the
amodal-height / truncation bias. A well-tuned multiplier (~1.0) trains the residual to convergence
and lowers depth error. Everything else is fixed.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — build_lr_mult() -> float
# ================================================================
def build_lr_mult():
    # WEAK DEFAULT: tiny LR multiplier -> the depth residual is essentially untrained.
    return 0.01
# ================================================================
# END EDITABLE REGION
# ================================================================
