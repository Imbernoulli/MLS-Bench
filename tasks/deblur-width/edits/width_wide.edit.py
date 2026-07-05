"""Strong width baseline for deblur-width (the good answer).

The strong reference: a WIDE backbone (32 base channels) -> more capacity, sharper. Reference: vendor/image-deblur/baselines/arch_width_wide.py
"""

_FILE = "image-deblur/solution/arch_width.py"

_CONTENT = '''def get_arch_config():
    # a WIDE backbone (32 base channels) -> more capacity, sharper
    return {"width": 32}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
