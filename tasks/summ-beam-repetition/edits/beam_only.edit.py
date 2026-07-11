"""Baseline `beam_only` for summ-beam-repetition.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/beam.py"

_CONTENT = '''def build_beam_config() -> dict:
    # Beam search alone, no n-gram block.
    return {"num_beams": 4, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
