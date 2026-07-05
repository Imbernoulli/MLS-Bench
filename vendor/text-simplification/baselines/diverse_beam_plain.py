"""Strong baseline: plain beam search, no grouping (num_beam_groups=1)."""


def build_diverse_beam_config() -> dict:
    return {"num_beam_groups": 1, "diversity_penalty": 0.0}
