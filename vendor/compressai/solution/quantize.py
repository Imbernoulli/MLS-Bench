"""Agent-editable quantization surrogate for the fixed compression pipeline.

Return one of ``none``, ``ste``, ``noise``, ``softround``, or ``ste_noise``.
Evaluation always uses the harness's fixed quantized path. Invalid values fail
verification instead of selecting a replacement surrogate.
"""
from __future__ import annotations


def quantize() -> str:
    return "none"
