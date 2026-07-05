"""Strong beam baseline: beam search (width 5) + no-repeat-3gram block."""


def build_beam_config() -> dict:
    return {"num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}
