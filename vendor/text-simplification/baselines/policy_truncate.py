"""Truncation baseline: keep the first 75% of the words (delete the tail).

A naive deletion heuristic — some SARI Delete credit, but no paraphrase/add and
arbitrary content dropped. Lands ABOVE copy-input and BELOW the real model."""


def build_policy() -> str:
    return "truncate"
