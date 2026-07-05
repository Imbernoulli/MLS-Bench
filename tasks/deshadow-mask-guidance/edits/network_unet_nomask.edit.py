"""Weak network baseline for deshadow-mask-guidance (the naive answer).

A BLIND U-Net that sees only the 3-channel shadowed RGB and must both LOCATE and correct the
shadow from colour alone (DeshadowNet-style multi-context net WITHOUT the mask prior). It
removes some shadow but, not knowing exactly where/how-much, leaks into the lit region and
mis-corrects the soft penumbra -> lower shadow-region PSNR than the mask-guided net.
Reference: vendor/image-deshadow/baselines/network_unet_nomask.py
"""

_FILE = "image-deshadow/solution/network.py"

_CONTENT = '''def get_network_config():
    # Blind U-Net (no mask input) -> must locate the shadow from RGB alone, lower shadow PSNR.
    return {"arch": "unet_nomask"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 39, "content": _CONTENT},
]
