"""STRONG mask baseline: mask-guided U-Net -- the soft shadow mask is concatenated as a 4th input channel (SP+M-Net, Le et al. ICCV 2019), telling the net exactly where/how-much to brighten -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/mask.py
"""

_FILE = "image-deshadow/solution/mask.py"

_CONTENT = 'def get_mask_config():\n    # Strong: mask-guided U-Net (mask as 4th channel), SP+M-Net recovery.\n    return {"use_mask": True}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 32, "end_line": 34, "content": _CONTENT},
]
