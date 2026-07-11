"""Strong baseline: standard no-repeat-3gram block."""
def build_norep_config() -> dict:
    return {"no_repeat_ngram_size": 3}
