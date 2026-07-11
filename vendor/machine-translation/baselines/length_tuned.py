"""Strong length baseline: tuned GNMT-style length penalty (~0.6 optimum)."""


def build_length_config() -> dict:
    return {"length_penalty": 0.6, "min_length": 0, "max_new_tokens": 128}
