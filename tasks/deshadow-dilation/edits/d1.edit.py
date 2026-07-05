"""WEAK dilation baseline: plain 3x3 bottleneck convs (dilation 1), small receptive field.
Reference: vendor/image-deshadow/solution/dilation.py
"""

_FILE = "image-deshadow/solution/dilation.py"

_CONTENT = 'def get_dilation_config():\n    # Weak: no dilation -> small receptive field, under-corrects large umbrae.\n    return {"dilations": [1, 1]}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 28, "end_line": 30, "content": _CONTENT},
]
