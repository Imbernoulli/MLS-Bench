"""Baseline `off` for summ-norepeat-ngram.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/norepeat.py"

_CONTENT = '''def build_norepeat_size() -> int:
    # No n-gram block (weak).
    return 0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 32, "end_line": 34, "content": _CONTENT},
]
