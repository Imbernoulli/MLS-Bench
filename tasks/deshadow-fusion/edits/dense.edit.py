"""STRONG fusion baseline: dense multi-level feature fusion -- concatenate features from every decoder level and fuse with a 1x1 conv (DenseNet/RDN-style) so coarse global-illumination and fine penumbra-edge features both feed the output -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/fusion.py
"""

_FILE = "image-deshadow/solution/fusion.py"

_CONTENT = 'def get_fusion_config():\n    # Strong: dense multi-level feature fusion (DenseNet/RDN-style).\n    return {"fusion": True}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
