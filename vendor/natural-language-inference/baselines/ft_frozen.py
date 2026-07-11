"""Candidate arm: frozen encoder with a trained classifier head."""
def build_finetune() -> dict:
    return {"encoder": "frozen"}
