"""Editable transform-family surface for ``flow-autoregressive-coupling``.

Return ``"affine"``, ``"maf"``, or ``"spline"``. The verifier holds depth,
width, base, and LU mixing fixed for a controlled transform-family ablation.
"""


# ================================================================
# EDITABLE REGION - select the conditioner family below
# ================================================================
def select_conditioner():
    return "maf"
# ================================================================
# END EDITABLE REGION
# ================================================================
