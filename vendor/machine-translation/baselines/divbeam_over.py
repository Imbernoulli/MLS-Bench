"""Degenerate baseline: 8 groups + large diversity penalty (off the MAP path)."""
def build_divbeam_config() -> dict:
    return {"num_beam_groups": 8, "diversity_penalty": 1.5}
