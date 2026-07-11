"""Baseline edit for nli-class-weighting: contradiction-focus class costs."""

_FILE = "natural-language-inference/solution/class_weighting.py"
_CONTENT = '''def build_weighting() -> dict:
    return {"weights": [0.75, 0.75, 1.5]}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
