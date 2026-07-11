"""Agent-editable residual-block choice for the fixed compression pipeline.

Return exactly ``{"residual": <bool>}``. The boolean selects the corresponding
fixed transform-stage implementation.
"""
from __future__ import annotations


def residual_design() -> dict:
    return {"residual": False}
