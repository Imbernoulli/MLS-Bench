"""Agent-editable synthesis upsampling choice for the fixed compression pipeline.

Return exactly one ``upsample_mode`` key with value ``nearest``, ``deconv``, or
``subpel``.
"""
from __future__ import annotations


def upsample_design() -> dict:
    return {"upsample_mode": "nearest"}
