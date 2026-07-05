"""Strong length baseline: sensibly compressive window (SARI-optimal)."""


def build_length_config() -> dict:
    return {"min_length": 0, "max_length": 96, "length_penalty": 1.0}
