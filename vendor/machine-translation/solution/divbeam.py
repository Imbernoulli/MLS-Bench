"""Diverse-beam configuration surface.

Return exactly `num_beam_groups` (a positive divisor of eight) and a finite
`diversity_penalty` in [0, 5]. The verifier parses the return literal statically.
"""
from __future__ import annotations


# EDITABLE REGION
def build_divbeam_config() -> dict:
    return {"num_beam_groups": 8, "diversity_penalty": 1.5}
# END EDITABLE REGION
