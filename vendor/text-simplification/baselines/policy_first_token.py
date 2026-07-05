"""Degenerate floor baseline: emit only the first source word (low SARI)."""


def build_policy() -> str:
    return "first_token"
