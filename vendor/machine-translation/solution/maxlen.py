"""Output-token-budget surface.

Return an integer from 1 through 160. The verifier parses the return literal
statically and does not execute this module.
"""
from __future__ import annotations


# EDITABLE REGION
def build_max_new_tokens() -> int:
    return 10
# END EDITABLE REGION
