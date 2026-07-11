"""Chance-floor baseline: majority-class predictor (~1/3)."""
def build_truncation() -> dict:
    return {"mode": "majority"}
