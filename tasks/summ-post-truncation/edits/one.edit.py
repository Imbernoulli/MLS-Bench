"""Baseline `one` for summ-post-truncation.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/posttrunc.py"

_CONTENT = '''def build_keep_sentences() -> int:
    # Retain one sentence.
    return 1'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
