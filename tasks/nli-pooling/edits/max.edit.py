"""Baseline edit for nli-pooling: attention-masked max pooling."""

_FILE = "natural-language-inference/solution/pooling.py"

_CONTENT = '''def build_pooling() -> dict:
    return {"pooling": "max"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 39, "content": _CONTENT},
]
