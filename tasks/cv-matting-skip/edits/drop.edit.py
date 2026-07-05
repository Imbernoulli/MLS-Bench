"""Weak baseline for cv-matting-skip: DROP-SKIP (discard the encoder skip).

Fills the expected concat width with zeros instead of the encoder skip, so no
encoder high-res detail reaches the decoder -> blurry matte -> high SAD. This is the
starting default in vendor/image-matting/solution/skip.py.
Reference: vendor/image-matting/baselines/skip_drop.py
"""

_FILE = "image-matting/solution/skip.py"

_CONTENT = '''def fuse(dec_up, skip):
    return torch.cat([dec_up, torch.zeros_like(skip)], 1)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 37, "content": _CONTENT},
]
