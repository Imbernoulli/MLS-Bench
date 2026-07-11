"""Candidate arm: increase the contradiction loss cost at fixed mean weight."""


def build_weighting() -> dict:
    return {"weights": [0.75, 0.75, 1.5]}
