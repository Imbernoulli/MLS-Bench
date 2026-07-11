"""Source-token truncation surface.

Return an integer from 1 through 128. The verifier parses the return literal
statically and does not execute this module.
"""
from __future__ import annotations


# EDITABLE REGION
def build_source_max_tokens() -> int:
    return 12
# END EDITABLE REGION
