"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/loss.py"

_CONTENT = '''def get_matting_loss():
    # Uniform whole-image alpha-L1 reference implementation.
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        return (pred - gt).abs().mean()
    return loss_fn'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 19, "content": _CONTENT},
]
