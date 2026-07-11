"""Translation decode-policy surface.

Return one of: "beam", "greedy", "copy_source", "first_token", or "empty".
The verifier parses the string literal statically.
"""
from __future__ import annotations


# EDITABLE REGION
def build_strategy() -> str:
    return "copy_source"
# END EDITABLE REGION
