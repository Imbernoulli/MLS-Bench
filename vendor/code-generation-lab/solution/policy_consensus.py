"""Agent-editable canonicalization policy for code self-consistency."""
from __future__ import annotations


def canonical(program):
    """Return a hashable key used to cluster one candidate program."""
    return (program or "").strip()
