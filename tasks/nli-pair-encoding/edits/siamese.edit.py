"""Candidate edit: siamese bi-encoder pair representation.
Reference: vendor/natural-language-inference/baselines/enc_siamese.py
"""

_FILE = "natural-language-inference/solution/pair_encoding.py"

_CONTENT = '''def build_encoding() -> dict:
    # Encode the two sentences separately.
    return {"encoding": "siamese"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 41, "content": _CONTENT},
]
