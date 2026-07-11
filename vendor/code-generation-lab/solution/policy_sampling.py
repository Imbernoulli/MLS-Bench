"""Agent-editable diversity policy for codegen-sampling-strategy.

The harness fixes the prompt, pool size, token cap, selection rule, model, and
problem inventory. Return only the temperature and nucleus cutoff used to draw
the fixed eight-candidate pool.
"""
from __future__ import annotations


def sampling_parameters(problem):
    """Return ``(temperature, top_p)`` for one policy-visible problem."""
    return 0.2, 0.95
