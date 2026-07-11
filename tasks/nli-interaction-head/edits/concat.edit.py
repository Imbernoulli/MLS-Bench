"""Candidate edit: concatenated [u; v] siamese interaction.
Reference: vendor/natural-language-inference/baselines/head_concat.py
"""

_FILE = "natural-language-inference/solution/interaction_head.py"

_CONTENT = '''def build_head() -> dict:
    # Use the concatenated sentence vectors.
    return {"interaction": "concat"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
