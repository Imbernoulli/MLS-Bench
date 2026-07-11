"""Chance-floor baseline: majority-class predictor (label-blind, ~1/3).
Reference: vendor/natural-language-inference/baselines/bias_majority.py
"""

_FILE = "natural-language-inference/solution/hypothesis_bias.py"

_CONTENT = '''def build_bias() -> dict:
    # Chance floor: label-blind majority-class predictor (~1/3 accuracy).
    return {"mode": "majority"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 40, "content": _CONTENT},
]
