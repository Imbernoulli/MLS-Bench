"""Good baseline for cv-matting-loss-design: unknown-band L1 + composition.

Restricts the alpha-L1 to the trimap UNKNOWN band and adds a composition term
|I - (a*F + (1-a)*B)| (Deep Image Matting, w=0.5), focusing capacity on the hard
transition -> lower SAD with clear headroom over the whole-image L1.
Reference: vendor/image-matting/baselines/loss_unk_comp.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 47, "end_line": 57, "content": _CONTENT},
]
