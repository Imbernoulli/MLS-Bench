"""Weak baseline: large min-length floor (forces padding past natural EOS)."""


def build_min_length() -> int:
    return 60
