"""Editable base-distribution surface for ``flow-base-distribution``.

Return ``"gaussian"``, ``"gaussian_trainable"``, or ``"gmm"``. The
minimal affine flow above the base is fixed by the verifier.
"""


# ================================================================
# EDITABLE REGION - select the base distribution below
# ================================================================
def select_base_distribution():
    return "gaussian"
# ================================================================
# END EDITABLE REGION
# ================================================================
