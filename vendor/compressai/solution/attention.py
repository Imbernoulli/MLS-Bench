"""Agent-editable attention choice for the fixed learned-compression pipeline.

Return exactly ``{"attention": <bool>}``. The boolean controls whether the
transform includes the fixed attention block implementation.
"""
from __future__ import annotations


def attention_design() -> dict:
    return {"attention": False}
