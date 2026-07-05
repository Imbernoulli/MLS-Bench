"""Strong loss baseline for deblur-loss-design (the good answer).

Optimise toward the TRUE SHARP target (`target_smooth=0`) with a robust Charbonnier loss
(Lai et al. LapSRN; used by MPRNet, Zamir et al. CVPR 2021) plus an EDGE (image-gradient)
term that rewards restoring high-frequency detail -> genuinely sharp restorations, high
deblur PSNR. Reference: vendor/image-deblur/baselines/loss_sharp.py
"""

_FILE = "image-deblur/solution/loss.py"

_CONTENT = '''def get_loss_config():
    # Sharp target (no over-smoothing) + Charbonnier + edge -> high deblur PSNR.
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 37, "content": _CONTENT},
]
