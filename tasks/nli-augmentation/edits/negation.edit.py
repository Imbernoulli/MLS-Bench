"""Baseline edit for nli-augmentation: negation.
Reference: vendor/natural-language-inference/baselines/ (see validate_all.py).
"""

_FILE = "natural-language-inference/solution/augmentation.py"

_CONTENT = '''def build_augment() -> dict:
    return {"augment": "negation"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
