"""WEAK residual baseline: regress the clean image directly (clean = net(.)) -- must reconstruct the whole scene from scratch, over-smooths, lower shadow-region PSNR.
Reference: vendor/image-deshadow/solution/residual.py
"""

_FILE = "image-deshadow/solution/residual.py"

_CONTENT = 'def get_residual_config():\n    # Weak: direct clean-image regression.\n    return {"mode": "direct"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 31, "content": _CONTENT},
]
