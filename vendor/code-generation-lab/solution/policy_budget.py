"""Agent-editable global candidate-allocation policy."""
from __future__ import annotations


def allocation_weights(problems):
    """Return one non-negative hardness weight for every problem."""
    return [1.0 for _problem in problems]
