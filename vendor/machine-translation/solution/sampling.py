"""Translation search-policy surface.

Return one of: "sample_t1", "topp", "greedy", or "beam".
The verifier parses the string literal statically.
"""
from __future__ import annotations


# EDITABLE REGION
def build_mode() -> str:
    return "sample_t1"
# END EDITABLE REGION
