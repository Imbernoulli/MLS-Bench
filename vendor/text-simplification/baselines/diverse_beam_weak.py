"""Weak baseline: many groups, strong diversity penalty (top-1 hurt)."""


def build_diverse_beam_config() -> dict:
    return {"num_beam_groups": 6, "diversity_penalty": 3.0}
