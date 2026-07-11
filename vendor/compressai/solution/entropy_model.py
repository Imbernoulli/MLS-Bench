"""Agent-editable entropy-model choice for the fixed compression pipeline.

Return one of ``factorized``, ``hyperprior_scale``, or ``meanscale``. Each value
selects a pinned official CompressAI model-zoo family. Invalid values fail
verification instead of selecting a different model.
"""
from __future__ import annotations


def entropy_model() -> str:
    return "factorized"
