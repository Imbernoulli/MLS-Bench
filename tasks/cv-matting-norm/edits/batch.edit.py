"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/norm.py"

_CONTENT = '''def make_norm(num_ch):
    return nn.BatchNorm2d(num_ch)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 16, "content": _CONTENT},
]
