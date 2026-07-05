"""Strong attention baseline for deblur-attention (the good answer).

The strong reference: channel attention ON (SE / MPRNet CAB) -> sharper. Reference: vendor/image-deblur/baselines/arch_attn_on.py
"""

_FILE = "image-deblur/solution/arch_attention.py"

_CONTENT = '''def get_arch_config():
    # channel attention ON (SE / MPRNet CAB) -> sharper
    return {"attention": True}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
