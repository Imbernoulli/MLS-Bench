"""Strong edge baseline for deblur-edge-loss (the good answer).

The strong reference: Charbonnier + a strong edge/gradient loss -> sharper edges, higher PSNR. Reference: vendor/image-deblur/baselines/loss_edge_on.py
"""

_FILE = "image-deblur/solution/edge.py"

_CONTENT = '''def get_loss_config():
    # Charbonnier + a strong edge/gradient loss -> sharper edges, higher PSNR
    return {"kind": "charbonnier", "edge_weight": 0.5, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
