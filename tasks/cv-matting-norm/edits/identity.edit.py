"""Weak baseline for cv-matting-norm: IDENTITY (no normalisation).

No normalisation after each conv -> the short fine-tune is slower / less stable ->
higher SAD. This is the starting default in vendor/image-matting/solution/norm.py.
Reference: vendor/image-matting/baselines/norm_identity.py
"""

_FILE = "image-matting/solution/norm.py"

_CONTENT = '''def make_norm(num_ch):
    return nn.Identity()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 34, "content": _CONTENT},
]
