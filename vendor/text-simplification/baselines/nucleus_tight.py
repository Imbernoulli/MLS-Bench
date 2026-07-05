"""Strong baseline: tight nucleus (restricted to the model's most probable tokens)."""


def build_top_p() -> float:
    return 0.6
