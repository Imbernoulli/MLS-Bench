"""Baseline `ngram3` for summ-norepeat-ngram.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/norepeat.py"

_CONTENT = '''def build_norepeat_size() -> int:
    # No-repeat-3gram block.
    return 3'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 32, "end_line": 34, "content": _CONTENT},
]
