"""Baseline `greedy` for summ-beam-repetition.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/beam.py"

_CONTENT = '''def build_beam_config() -> dict:
    # Greedy decoding, no repetition control (the weak default).
    return {"num_beams": 1, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
