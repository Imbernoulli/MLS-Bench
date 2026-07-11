"""Baseline `beam` for summ-sampling-vs-beam.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/sampling.py"

_CONTENT = '''def build_decode_strategy() -> dict:
    # Deterministic four-beam decoding.
    return {"strategy": "beam", "num_beams": 4}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
