"""Weak length baseline: runaway-long window -> under-compressed (near copy-input)."""


def build_length_config() -> dict:
    return {"min_length": 40, "max_length": 160, "length_penalty": 2.5}
