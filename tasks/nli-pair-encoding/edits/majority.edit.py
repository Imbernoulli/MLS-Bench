"""Chance-floor baseline: majority-class predictor (label-blind, ~1/3).
Reference: vendor/natural-language-inference/baselines/enc_majority.py
"""

_FILE = "natural-language-inference/solution/pair_encoding.py"

_CONTENT = '''def build_encoding() -> dict:
    # Chance floor: label-blind majority-class predictor (~1/3 accuracy).
    return {"encoding": "majority"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 41, "content": _CONTENT},
]
