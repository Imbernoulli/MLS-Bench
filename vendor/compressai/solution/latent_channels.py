"""Agent-editable latent width for a fixed learned-compression pipeline.

Return exactly ``{"M": <integer>}``, where ``M`` is in ``[8, 192]``.
"""
from __future__ import annotations


def latent_design() -> dict:
    return {"M": 16}
