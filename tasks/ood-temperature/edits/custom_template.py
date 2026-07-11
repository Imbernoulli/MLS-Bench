"""Editable energy-temperature surface for ``ood-temperature``.

Return a finite temperature in ``[1e-3, 1e4]``. The energy formula and all
classifier representations are fixed by the verifier.
"""


# ================================================================
# EDITABLE REGION - select the energy temperature below
# ================================================================
def select_temperature():
    return 1000.0
# ================================================================
# END EDITABLE REGION
# ================================================================
