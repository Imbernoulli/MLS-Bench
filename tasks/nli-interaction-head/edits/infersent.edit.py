"""Candidate edit: InferSent features [u; v; |u-v|; u*v].
Reference: vendor/natural-language-inference/baselines/head_infersent.py
"""

_FILE = "natural-language-inference/solution/interaction_head.py"

_CONTENT = '''def build_head() -> dict:
    # Use InferSent matching features (Conneau 2017).
    return {"interaction": "infersent"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
