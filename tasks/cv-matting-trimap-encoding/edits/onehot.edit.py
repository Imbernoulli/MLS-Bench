"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/trimap.py"
_CONTENT = '''def encode_trimap(trimap):
    foreground = (trimap >= 0.75).float()
    background = (trimap <= 0.25).float()
    unknown = 1.0 - foreground - background
    return torch.stack((foreground, unknown, background), dim=1)'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 14, "content": _CONTENT},
]
