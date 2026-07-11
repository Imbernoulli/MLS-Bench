"""Weak beam baseline: greedy decoding, no repetition control."""


def build_beam_config() -> dict:
    return {"num_beams": 1, "no_repeat_ngram_size": 0}
