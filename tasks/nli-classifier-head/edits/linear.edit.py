"""Baseline edit for nli-classifier-head: linear.
Reference: vendor/natural-language-inference/baselines/ (see validate_all.py).
"""

_FILE = "natural-language-inference/solution/classifier_head.py"

_CONTENT = '''def build_classifier() -> dict:
    return {"head": "linear"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
