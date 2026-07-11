"""Choose codec families by quality band for the fixed bitrate schedule."""
from __future__ import annotations


def bitrate_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
