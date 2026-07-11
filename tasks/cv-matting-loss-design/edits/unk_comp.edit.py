"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/loss.py"

_CONTENT = '''def get_matting_loss():
    # Unknown-band L1 + composition (Deep Image Matting): focus on the transition.
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        u = unknown.float()
        d = u.sum(dim=(-2, -1)).clamp(min=1.0)
        alpha_l = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / d
        comp = pred.unsqueeze(1) * fg + (1 - pred.unsqueeze(1)) * bg
        comp_l = ((comp - image).abs().mean(1) * u).sum(dim=(-2, -1)) / d
        return (alpha_l + 0.5 * comp_l).mean()
    return loss_fn'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 19, "content": _CONTENT},
]
