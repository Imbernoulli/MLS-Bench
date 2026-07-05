"""WEAK mask baseline: blind U-Net that sees only the 3-ch shadowed RGB (DeshadowNet, Qu et al. CVPR 2017) -- must locate the shadow from colour alone, lower shadow-region PSNR.
Reference: vendor/image-deshadow/solution/mask.py
"""

_FILE = "image-deshadow/solution/mask.py"

_CONTENT = 'def get_mask_config():\n    # Weak: blind U-Net, no mask channel (DeshadowNet-style).\n    return {"use_mask": False}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 32, "end_line": 34, "content": _CONTENT},
]
