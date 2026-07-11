"""Candidate arm: increase the neutral-class loss cost at fixed mean weight."""


def build_weighting() -> dict:
    return {"weights": [0.75, 1.5, 0.75]}
