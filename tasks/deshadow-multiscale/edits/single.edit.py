"""WEAK multiscale baseline: a single-scale mask-guided U-Net.
Reference: vendor/image-deshadow/solution/multiscale.py
"""

_FILE = "image-deshadow/solution/multiscale.py"

_CONTENT = 'def get_multiscale_config():\n    # Weak: single-scale mask-guided U-Net.\n    return {"multiscale": False}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 27, "end_line": 29, "content": _CONTENT},
]
