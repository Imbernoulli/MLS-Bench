"""Baseline `all` for summ-post-truncation.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/posttrunc.py"

_CONTENT = '''def build_keep_sentences() -> int:
    # Keep the full decoded summary (strong).
    return 999'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
