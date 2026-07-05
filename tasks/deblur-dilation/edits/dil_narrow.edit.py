"""Weak dilation baseline for deblur-dilation (the naive answer).

The naive / degenerate choice: dilation=1 (narrow receptive field) -> cannot cover large blur. Reference: vendor/image-deblur/baselines/arch_dil_narrow.py
"""

_FILE = "image-deblur/solution/arch_dilation.py"

_CONTENT = '''def get_arch_config():
    # dilation=1 (narrow receptive field) -> cannot cover large blur
    return {"dilation": 1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
