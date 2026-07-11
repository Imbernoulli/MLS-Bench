"""Chance-floor baseline: majority-class predictor (~1/3)."""
def build_finetune() -> dict:
    return {"encoder": "majority"}
