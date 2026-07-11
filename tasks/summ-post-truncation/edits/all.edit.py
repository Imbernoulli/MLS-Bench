"""Baseline `all` for summ-post-truncation.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/posttrunc.py"

_CONTENT = '''def build_keep_sentences() -> int:
    # Retain up to 999 sentences.
    return 999'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
