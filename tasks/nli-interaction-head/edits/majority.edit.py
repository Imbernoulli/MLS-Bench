"""Chance-floor baseline: majority-class predictor (label-blind, ~1/3).
Reference: vendor/natural-language-inference/baselines/head_majority.py
"""

_FILE = "natural-language-inference/solution/interaction_head.py"

_CONTENT = '''def build_head() -> dict:
    # Chance floor: label-blind majority-class predictor (~1/3 accuracy).
    return {"interaction": "majority"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
