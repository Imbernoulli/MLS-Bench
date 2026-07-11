"""Chance-floor baseline: majority-class predictor (~1/3)."""
def build_augment() -> dict:
    return {"augment": "majority"}
