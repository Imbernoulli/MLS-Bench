"""Map low-, mid-, and high-texture content to pinned codec families."""
from __future__ import annotations


def content_policy() -> tuple[str, str, str]:
    return ("factorized", "factorized", "factorized")
