"""Chance-floor baseline: majority-class predictor (label-blind, ~1/3)."""
def build_head() -> dict:
    return {"interaction": "majority"}
