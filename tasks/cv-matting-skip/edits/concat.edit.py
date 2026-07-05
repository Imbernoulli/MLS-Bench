"""SOTA baseline for cv-matting-skip: FULL concat skip (standard U-Net fusion).

Concatenate the full-strength encoder skip with the decoder feature (standard U-Net
fusion), injecting the encoder's high-resolution detail into the decoder so the matte
keeps sharp boundaries -> lowest SAD with clear headroom over drop-skip.
Reference: vendor/image-matting/baselines/skip_concat.py
"""

_FILE = "image-matting/solution/skip.py"

_CONTENT = '''def fuse(dec_up, skip):
    return torch.cat([dec_up, skip], 1)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 37, "content": _CONTENT},
]
