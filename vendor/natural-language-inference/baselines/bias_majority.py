"""Chance-floor baseline: majority-class predictor on the HARD subset (~1/3)."""
def build_bias() -> dict:
    return {"mode": "majority"}
