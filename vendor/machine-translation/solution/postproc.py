"""Translation output post-processing surface.

Return one of: "identity", "normalize", "lowercase", or "strip_punct".
The verifier parses the string literal statically.
"""
from __future__ import annotations


# EDITABLE REGION
def build_postproc() -> str:
    return "lowercase"
# END EDITABLE REGION
