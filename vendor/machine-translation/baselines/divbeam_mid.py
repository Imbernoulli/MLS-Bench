"""Middle baseline: 4 groups + small diversity penalty."""
def build_divbeam_config() -> dict:
    return {"num_beam_groups": 4, "diversity_penalty": 0.5}
