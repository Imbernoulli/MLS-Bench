"""Chance-floor baseline: majority-class predictor (~1/3)."""
def build_reg() -> dict:
    return {"reg": "majority"}
