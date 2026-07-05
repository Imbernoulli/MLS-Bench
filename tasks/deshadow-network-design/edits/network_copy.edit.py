"""Weak network baseline for deshadow-network-design (the naive answer = the default).

Pass the shadowed input straight through, NO removal. The do-nothing floor: it scores exactly
the shadowed-input SHADOW-REGION PSNR, so any real deshadower must beat it.
Reference: vendor/image-deshadow/baselines/network_copy.py
"""

_FILE = "image-deshadow/solution/network.py"

_CONTENT = '''def get_network_config():
    # Copy the shadowed input through (no removal) -> the do-nothing floor.
    return {"arch": "copy"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 39, "content": _CONTENT},
]
