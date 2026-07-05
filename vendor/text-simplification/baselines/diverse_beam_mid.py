"""Mid baseline: 2 groups, moderate diversity penalty."""


def build_diverse_beam_config() -> dict:
    return {"num_beam_groups": 2, "diversity_penalty": 0.5}
