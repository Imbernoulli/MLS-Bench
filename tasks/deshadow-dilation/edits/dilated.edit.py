"""STRONG dilation baseline: dilated bottleneck convs (rates 2,4) -> large receptive field (ASPP/multi-context-deshadow style), covers big soft shadows -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/dilation.py
"""

_FILE = "image-deshadow/solution/dilation.py"

_CONTENT = 'def get_dilation_config():\n    # Strong: dilated bottleneck -> large receptive field covers big shadows.\n    return {"dilations": [2, 4]}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 28, "end_line": 30, "content": _CONTENT},
]
