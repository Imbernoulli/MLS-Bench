"""Strong network baseline for deshadow-mask-guidance (the good answer = SOTA reference).

The MASK-GUIDED U-Net: the soft shadow mask is concatenated as a 4th input channel, so the
net knows exactly WHERE and HOW MUCH to brighten -- the SP+M-Net (Le et al. ICCV 2019)
physically-parameterised recovery that fits the multiplicative attenuation I = a*J. Highest
shadow-region PSNR.
Reference: vendor/image-deshadow/baselines/network_unet_mask.py
"""

_FILE = "image-deshadow/solution/network.py"

_CONTENT = '''def get_network_config():
    # Mask-guided U-Net (shadow mask as 4th input) -> knows where/how-much to brighten, higher shadow PSNR.
    return {"arch": "unet_mask"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 39, "content": _CONTENT},
]
