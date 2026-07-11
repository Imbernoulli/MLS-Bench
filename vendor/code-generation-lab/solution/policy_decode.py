"""Agent-editable global generation-length allocation policy."""
from __future__ import annotations


def token_cap_weights(problems):
    """Return one non-negative length-need weight for every problem."""
    return [1.0 for _problem in problems]
