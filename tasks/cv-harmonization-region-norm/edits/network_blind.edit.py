"""Mask-blind baseline for cv-harmonization-region-norm (region-agnostic weak learned).

A MASK-BLIND encoder-decoder U-Net (composite RGB only): a region-agnostic image-to-image
net that CANNOT tell the pasted foreground from the background, so it applies a compromise
correction that disturbs the already-correct background and only partially fixes the
foreground -> a middling foreground PSNR (above the do-nothing floor, below the
mask-conditioned net).
Reference: vendor/image-harmonization/baselines/network_blind.py
"""

_FILE = "image-harmonization/solution/network.py"

_CONTENT = '''def get_network_config():
    # Mask-BLIND region-agnostic U-Net -> partial recovery only.
    return {"arch": "blind"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
