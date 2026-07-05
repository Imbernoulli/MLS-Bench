"""STRONG upsampling baseline: bilinear-resize + conv upsampling -- smooth, artifact-free, respects the soft penumbra falloff -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/upsampling.py
"""

_FILE = "image-deshadow/solution/upsampling.py"

_CONTENT = 'def get_upsampling_config():\n    # Strong: bilinear-resize + conv (artifact-free) upsampling.\n    return {"up": "bilinear"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
