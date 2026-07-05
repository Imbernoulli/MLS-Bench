"""STRONG multiscale baseline: a coarse-to-fine pyramid that relights at half resolution first (large effective receptive field, captures the whole big soft shadow) then refines at full resolution (MSPFN/pyramid restoration style) -> higher shadow-region PSNR on large shadows.
Reference: vendor/image-deshadow/solution/multiscale.py
"""

_FILE = "image-deshadow/solution/multiscale.py"

_CONTENT = 'def get_multiscale_config():\n    # Strong: coarse-to-fine pyramid (relight at half-res then refine).\n    return {"multiscale": True}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 27, "end_line": 29, "content": _CONTENT},
]
