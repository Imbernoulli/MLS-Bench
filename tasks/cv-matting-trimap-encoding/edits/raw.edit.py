"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/trimap.py"
_CONTENT = '''def encode_trimap(trimap):
    return trimap.unsqueeze(1)'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 14, "content": _CONTENT},
]
