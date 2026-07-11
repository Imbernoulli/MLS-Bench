"""Agent-editable rate-distortion controls for the fixed compression pipeline.

Supported keys are ``lmbda`` in ``[1e-4, 1.0]``, ``target_bpp`` as ``None`` or a
positive finite number, and ``rate_gain`` in ``[0, 10]``. Unknown or malformed
values fail verification.
"""
from __future__ import annotations


def rd_control() -> dict:
    return {"lmbda": 0.002, "target_bpp": None, "rate_gain": 1.0}
