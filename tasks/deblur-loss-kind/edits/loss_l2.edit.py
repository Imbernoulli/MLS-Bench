"""Weak loss baseline for deblur-loss-kind (the naive answer).

The naive / degenerate choice: plain L2 (MSE), no edge term -> over-smoothed, lower deblur PSNR. Reference: vendor/image-deblur/baselines/loss_l2.py
"""

_FILE = "image-deblur/solution/losskind.py"

_CONTENT = '''def get_loss_config():
    # plain L2 (MSE), no edge term -> over-smoothed, lower deblur PSNR
    return {"kind": "l2", "edge_weight": 0.0, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
