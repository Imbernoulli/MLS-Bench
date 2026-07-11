"""Strong baseline: plain beam (1 group, no diversity)."""
def build_divbeam_config() -> dict:
    return {"num_beam_groups": 1, "diversity_penalty": 0.0}
