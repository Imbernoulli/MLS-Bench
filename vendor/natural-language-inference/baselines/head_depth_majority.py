"""Chance-floor baseline: majority-class predictor (~1/3)."""
def build_classifier() -> dict:
    return {"head": "majority"}
