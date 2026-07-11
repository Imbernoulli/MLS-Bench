"""Agent-editable normalization choice for the fixed compression pipeline.

Return exactly one ``norm`` key with value ``none`` or ``batchnorm``.
"""
from __future__ import annotations


def norm_design() -> dict:
    return {"norm": "batchnorm"}
