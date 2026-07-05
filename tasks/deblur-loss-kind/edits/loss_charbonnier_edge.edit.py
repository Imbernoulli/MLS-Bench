"""Strong loss baseline for deblur-loss-kind (the good answer).

The strong reference: Charbonnier + edge term -> sharp restorations, higher PSNR. Reference: vendor/image-deblur/baselines/loss_charbonnier_edge.py
"""

_FILE = "image-deblur/solution/losskind.py"

_CONTENT = '''def get_loss_config():
    # Charbonnier + edge term -> sharp restorations, higher PSNR
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
