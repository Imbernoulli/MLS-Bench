"""Weak length baseline: strongly LONG-biased length penalty -> over-generation."""


def build_length_config() -> dict:
    return {"length_penalty": 2.0, "min_length": 0, "max_new_tokens": 128}
