"""Agent-editable activation choice for the fixed learned-compression pipeline.

Return exactly one key, ``activation``, with value ``identity``, ``relu``, or
``gdn``. The harness evaluates the resulting trained codec; it does not replace
invalid values with another implementation.
"""
from __future__ import annotations


def activation_design() -> dict:
    return {"activation": "identity"}
