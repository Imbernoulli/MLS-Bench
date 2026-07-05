"""Weak loss baseline for deblur-loss-design (the naive answer = the default).

Optimise toward an OVER-SMOOTHED (Gaussian low-pass) target (`target_smooth=1.2`): the
network learns to reproduce a blurred version of the ground truth -- the classic
L2-conditional-mean over-smoothing failure -- so it throws away the high-frequency detail
it should restore and deblur PSNR collapses (often below the blurry-input floor).
Reference: vendor/image-deblur/baselines/loss_smoothed.py
"""

_FILE = "image-deblur/solution/loss.py"

_CONTENT = '''def get_loss_config():
    # Over-smoothed target -> the net reproduces blur, low deblur PSNR.
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 1.2}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 37, "content": _CONTENT},
]
