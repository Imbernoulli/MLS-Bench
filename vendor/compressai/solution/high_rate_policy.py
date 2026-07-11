"""Choose codec families for three quality bands under a high-rate objective."""
from __future__ import annotations


def high_rate_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
