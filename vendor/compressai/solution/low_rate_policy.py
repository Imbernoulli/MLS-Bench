"""Choose codec families for three quality bands under a low-rate objective."""
from __future__ import annotations


def low_rate_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
