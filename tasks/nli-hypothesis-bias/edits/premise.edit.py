"""Candidate edit: full pair using the premise (cross-encoder).
Reference: vendor/natural-language-inference/baselines/bias_premise.py
"""

_FILE = "natural-language-inference/solution/hypothesis_bias.py"

_CONTENT = '''def build_bias() -> dict:
    # Supply both the premise and hypothesis.
    return {"use_premise": True}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
