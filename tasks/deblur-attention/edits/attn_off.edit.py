"""Weak attention baseline for deblur-attention (the naive answer).

The naive / degenerate choice: NO channel attention (plain ResBlocks). Reference: vendor/image-deblur/baselines/arch_attn_off.py
"""

_FILE = "image-deblur/solution/arch_attention.py"

_CONTENT = '''def get_arch_config():
    # NO channel attention (plain ResBlocks)
    return {"attention": False}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
