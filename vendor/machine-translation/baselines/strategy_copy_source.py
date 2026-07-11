"""Degenerate strategy baseline: copy the German source (~0 BLEU)."""


def build_strategy() -> str:
    return "copy_source"
