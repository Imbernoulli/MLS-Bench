"""Token repetition-penalty surface.

Return exactly `repetition_penalty` as a finite number in [0.1, 5].
The verifier parses the mapping literal statically.
"""
from __future__ import annotations


# EDITABLE REGION
def build_reppen_config() -> dict:
    return {"repetition_penalty": 1.0}
# END EDITABLE REGION
