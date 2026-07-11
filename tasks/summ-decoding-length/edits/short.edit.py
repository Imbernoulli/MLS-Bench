"""Baseline `short` for summ-decoding-length.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/length.py"

_CONTENT = '''def build_length_config() -> dict:
    # Degenerate-short: clipped summaries, low ROUGE recall.
    return {"min_length": 1, "max_length": 20, "length_penalty": 0.2}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 42, "content": _CONTENT},
]
