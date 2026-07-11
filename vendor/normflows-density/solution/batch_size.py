"""Editable batch-size surface for ``flow-batch-size``.

Return an integer batch size in ``[1, 8192]``. Architecture, optimizer, learning
rate, data, and step budget are fixed by the verifier.
"""


# ================================================================
# EDITABLE REGION - select the training batch size below
# ================================================================
def select_batch_size():
    return 8
# ================================================================
# END EDITABLE REGION
# ================================================================
