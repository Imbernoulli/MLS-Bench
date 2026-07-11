"""Baseline `long` for summ-decoding-length.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/length.py"

_CONTENT = '''def build_length_config() -> dict:
    # Runaway-long: inflates recall but precision collapses under F1.
    return {"min_length": 150, "max_length": 200, "length_penalty": 3.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 42, "content": _CONTENT},
]
