"""Candidate arm: 128-token joint sequence cap."""
def build_truncation() -> dict:
    return {"max_len": 128}
