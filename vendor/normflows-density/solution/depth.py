"""Editable flow-depth surface for ``flow-depth-permutation``.

Return an integer number of affine coupling layers in ``[1, 32]``. The
permutation is fixed to a deterministic swap because the current benchmark
anchors vary depth only.
"""


# ================================================================
# EDITABLE REGION - select the flow depth below
# ================================================================
def select_depth():
    return 2
# ================================================================
# END EDITABLE REGION
# ================================================================
