"""Candidate edit: cross-encoder pair representation.
Reference: vendor/natural-language-inference/baselines/enc_cross.py
"""

_FILE = "natural-language-inference/solution/pair_encoding.py"

_CONTENT = '''def build_encoding() -> dict:
    # Jointly encode premise and hypothesis tokens.
    return {"encoding": "cross"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 41, "content": _CONTENT},
]
