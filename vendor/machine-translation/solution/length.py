"""Beam length-normalization surface.

Return exactly `length_penalty`, `min_length`, and `max_new_tokens` as a
literal mapping within the ranges documented by the task.
"""
from __future__ import annotations


# EDITABLE REGION
def build_length_config() -> dict:
    return {"length_penalty": 2.0, "min_length": 0, "max_new_tokens": 128}
# END EDITABLE REGION
