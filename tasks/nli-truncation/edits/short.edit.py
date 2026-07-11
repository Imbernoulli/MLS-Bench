"""Baseline edit for nli-truncation: short.
Reference: vendor/natural-language-inference/baselines/ (see validate_all.py).
"""

_FILE = "natural-language-inference/solution/truncation.py"

_CONTENT = '''def build_truncation() -> dict:
    return {"max_len": 16}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
