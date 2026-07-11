"""Agent-editable entropy-context choice for the fixed compression pipeline.

Return exactly ``{"use_context": <bool>}``. The boolean selects between the
fixed entropy-model variants implemented by the harness.
"""
from __future__ import annotations


def context_design() -> dict:
    return {"use_context": False}
