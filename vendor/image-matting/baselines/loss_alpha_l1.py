"""Weak baseline (negative control) for cv-matting-loss-design: plain alpha-L1.

The alpha-prediction loss only (L1 on alpha in the unknown band). No composition or
Laplacian term -> the matte structure is under-constrained -> high SAD. This is the
starting default in vendor/image-matting/solution/loss.py.
"""


def get_matting_loss():
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        u = unknown.float()
        d = u.sum(dim=(-2, -1)).clamp(min=1.0)
        alpha_l = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / d
        return alpha_l.mean()
    return loss_fn
