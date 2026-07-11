"""Choose one pinned codec family under the entropy-stream budget."""
from __future__ import annotations


def stream_budget_policy() -> str:
    return "factorized"
