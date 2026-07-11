"""Weak baseline: over-aggressive 1-gram repetition block."""
def build_norep_config() -> dict:
    return {"no_repeat_ngram_size": 1}
