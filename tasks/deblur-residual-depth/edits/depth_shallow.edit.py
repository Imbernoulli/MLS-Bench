"""Weak depth baseline for deblur-residual-depth (the naive answer).

The naive / degenerate choice: a SHALLOW net (1 ResBlock per stage) -> under-fits heavy blur. Reference: vendor/image-deblur/baselines/arch_depth_shallow.py
"""

_FILE = "image-deblur/solution/arch_depth.py"

_CONTENT = '''def get_arch_config():
    # a SHALLOW net (1 ResBlock per stage) -> under-fits heavy blur
    return {"n_resblocks": 1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
