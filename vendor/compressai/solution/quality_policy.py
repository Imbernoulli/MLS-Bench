"""Map low-, mid-, and high-quality bands to pinned codec families."""
from __future__ import annotations


def quality_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
