"""STRONG physics baseline: the SP+M-Net illumination model -- the net predicts per-pixel affine relighting parameters (w,b) and outputs J = w*I + b, a valid multiplicative-illumination inverse matching the true degradation form -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/physics.py
"""

_FILE = "image-deshadow/solution/physics.py"

_CONTENT = 'def get_physics_config():\n    # Strong: SP+M-Net affine illumination model J = w*I + b.\n    return {"mode": "physics"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 32, "content": _CONTENT},
]
