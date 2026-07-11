"""Baseline edit for nli-class-weighting: uniform.
Reference: vendor/natural-language-inference/baselines/ (see validate_all.py).
"""

_FILE = "natural-language-inference/solution/class_weighting.py"

_CONTENT = '''def build_weighting() -> dict:
    return {"weights": [1.0, 1.0, 1.0]}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
