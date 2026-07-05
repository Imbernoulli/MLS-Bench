"""STRONG residual baseline: residual learning -- clean = shadowed + net(.), predict only the shadow correction (an easy target for the near-multiplicative degradation) -> higher shadow-region PSNR.
Reference: vendor/image-deshadow/solution/residual.py
"""

_FILE = "image-deshadow/solution/residual.py"

_CONTENT = 'def get_residual_config():\n    # Strong: residual correction learning (clean = shadowed + net).\n    return {"mode": "residual"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 31, "content": _CONTENT},
]
