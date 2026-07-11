"""Choose codec families by quality band for the stated R-D objective."""
from __future__ import annotations


def objective_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
