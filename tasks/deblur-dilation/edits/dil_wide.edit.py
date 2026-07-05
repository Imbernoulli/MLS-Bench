"""Strong dilation baseline for deblur-dilation (the good answer).

The strong reference: dilation=4 (wide receptive field) -> covers larger streak, sharper. Reference: vendor/image-deblur/baselines/dil_wide.py
"""

_FILE = "image-deblur/solution/arch_dilation.py"

_CONTENT = '''def get_arch_config():
    # dilation=4 (wide receptive field) -> covers larger streak, sharper
    return {"dilation": 4}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
