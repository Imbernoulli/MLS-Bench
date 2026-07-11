"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/refine.py"
_CONTENT = '''def refine(coarse_alpha, image, trimap):
    return coarse_alpha'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 14, "content": _CONTENT},
]
