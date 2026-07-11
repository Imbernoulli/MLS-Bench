"""Editable Adam learning-rate surface for ``flow-learning-rate``.

Return a finite number in ``[1e-6, 1]``. The complete flow, batch size,
optimizer family, data, and step budget are fixed by the verifier.
"""


# ================================================================
# EDITABLE REGION - select the Adam learning rate below
# ================================================================
def select_learning_rate():
    return 5e-2
# ================================================================
# END EDITABLE REGION
# ================================================================
