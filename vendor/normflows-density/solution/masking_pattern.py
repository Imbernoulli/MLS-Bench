"""Editable RealNVP mask surface for ``flow-masking-pattern``.

Return exactly eight binary length-two masks. Each mask must contain both a zero
and a one. The transform, width, base distribution, and optimizer are fixed.
"""


# ================================================================
# EDITABLE REGION - select the eight binary masks below
# ================================================================
def select_masks():
    return [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
# ================================================================
# END EDITABLE REGION
# ================================================================
