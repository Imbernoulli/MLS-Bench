"""Beam-search configuration surface.

Return exactly `num_beams` (integer 1..12) and `no_repeat_ngram_size`
(integer 0..10). The verifier reads this function as a static literal and does not
execute this module.
"""
from __future__ import annotations


# EDITABLE REGION
def build_beam_config() -> dict:
    return {"num_beams": 1, "no_repeat_ngram_size": 0}
# END EDITABLE REGION
