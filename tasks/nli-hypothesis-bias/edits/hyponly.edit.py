"""Candidate edit: hypothesis-only input (premise masked out).
Reference: vendor/natural-language-inference/baselines/bias_hyponly.py
"""

_FILE = "natural-language-inference/solution/hypothesis_bias.py"

_CONTENT = '''def build_bias() -> dict:
    # Mask the premise and retain the hypothesis.
    return {"use_premise": False}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
