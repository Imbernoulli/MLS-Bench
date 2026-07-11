"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/skip.py"

_CONTENT = '''def fuse(dec_up, skip):
    return torch.cat([dec_up, skip], 1)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
