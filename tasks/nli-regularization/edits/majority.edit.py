"""Baseline edit for nli-regularization: majority.
Reference: vendor/natural-language-inference/baselines/ (see validate_all.py).
"""

_FILE = "natural-language-inference/solution/regularization.py"

_CONTENT = '''def build_reg() -> dict:
    return {"reg": "majority"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
