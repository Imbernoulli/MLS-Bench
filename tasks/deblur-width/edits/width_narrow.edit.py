"""Weak width baseline for deblur-width (the naive answer).

The naive / degenerate choice: a NARROW backbone (12 base channels) -> under-fits, lower deblur PSNR. Reference: vendor/image-deblur/baselines/arch_width_narrow.py
"""

_FILE = "image-deblur/solution/arch_width.py"

_CONTENT = '''def get_arch_config():
    # a NARROW backbone (12 base channels) -> under-fits, lower deblur PSNR
    return {"width": 12}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
