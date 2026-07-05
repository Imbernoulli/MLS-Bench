"""WEAK upsampling baseline: transpose-convolution (deconv) upsampling, prone to checkerboard artifacts across the smooth soft shadow.
Reference: vendor/image-deshadow/solution/upsampling.py
"""

_FILE = "image-deshadow/solution/upsampling.py"

_CONTENT = 'def get_upsampling_config():\n    # Weak: transpose-conv (checkerboard-prone) upsampling.\n    return {"up": "transpose"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
