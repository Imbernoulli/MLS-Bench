"""WEAK fusion baseline: use only the last decoder block's features.
Reference: vendor/image-deshadow/solution/fusion.py
"""

_FILE = "image-deshadow/solution/fusion.py"

_CONTENT = 'def get_fusion_config():\n    # Weak: last decoder block features only.\n    return {"fusion": False}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
