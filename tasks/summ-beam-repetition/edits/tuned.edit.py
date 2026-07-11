"""Baseline `tuned` for summ-beam-repetition.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/beam.py"

_CONTENT = '''def build_beam_config() -> dict:
    # Four-beam decoding with a no-repeat-3gram block.
    return {"num_beams": 4, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
