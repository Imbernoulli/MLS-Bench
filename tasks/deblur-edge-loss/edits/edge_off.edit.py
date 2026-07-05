"""Weak edge baseline for deblur-edge-loss (the naive answer).

The naive / degenerate choice: NO edge/gradient loss term -> edges under-restored. Reference: vendor/image-deblur/baselines/loss_edge_off.py
"""

_FILE = "image-deblur/solution/edge.py"

_CONTENT = '''def get_loss_config():
    # NO edge/gradient loss term -> edges under-restored
    return {"kind": "charbonnier", "edge_weight": 0.0, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
