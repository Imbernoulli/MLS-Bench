"""Editable conditioner-width surface for ``flow-conditioner-width``.

Return an integer hidden width in ``[2, 512]``. The coupling family, number of
layers, base distribution, and optimizer are fixed by the verifier.
"""


# ================================================================
# EDITABLE REGION - select the conditioner width below
# ================================================================
def select_conditioner_width():
    return 4
# ================================================================
# END EDITABLE REGION
# ================================================================
