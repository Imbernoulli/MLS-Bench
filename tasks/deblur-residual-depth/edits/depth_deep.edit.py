"""Strong depth baseline for deblur-residual-depth (the good answer).

The strong reference: a DEEPER stack (3 ResBlocks per stage) -> more capacity, sharper. Reference: vendor/image-deblur/baselines/arch_depth_deep.py
"""

_FILE = "image-deblur/solution/arch_depth.py"

_CONTENT = '''def get_arch_config():
    # a DEEPER stack (3 ResBlocks per stage) -> more capacity, sharper
    return {"n_resblocks": 3}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
