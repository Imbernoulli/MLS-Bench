"""Agent-editable transform width for the fixed compression pipeline.

Return exactly ``{"N": <integer>}``, where ``N`` is in ``[8, 128]``. Other
model axes remain fixed by the harness.
"""
from __future__ import annotations


def width_design() -> dict:
    return {"N": 16}
