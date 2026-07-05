"""Weak baseline (negative control) for cv-matting-loss-design: whole-image alpha-L1.

Averages |pred - gt| over EVERY pixel (fg + bg + unknown); the trivial solid regions
dominate the mean so the hard UNKNOWN transition is under-fit -> high SAD. This is the
starting default in vendor/image-matting/solution/loss.py.
Reference: vendor/image-matting/baselines/loss_whole_l1.py
"""

_FILE = "image-matting/solution/loss.py"

_CONTENT = '''def get_matting_loss():
    # Default: uniform whole-image alpha-L1 -> the hard unknown band is under-fit.
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        return (pred - gt).abs().mean()
    return loss_fn'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 47, "end_line": 57, "content": _CONTENT},
]
