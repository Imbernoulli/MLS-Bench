"""Good baseline for cv-matting-loss-design: unknown-band L1 + composition.

Restricts the alpha-L1 to the trimap UNKNOWN band (where matting is actually solved)
and adds a composition term |I - (a*F + (1-a)*B)| (Deep Image Matting, Xu et al.
2017, w=0.5), so the optimiser spends its capacity on the hard transition rather than
the trivial solid regions -> lower SAD with clear headroom over the uniform
whole-image L1. Reference: vendor/image-matting/baselines/loss_unk_comp.py
"""


def get_matting_loss():
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        u = unknown.float()
        d = u.sum(dim=(-2, -1)).clamp(min=1.0)
        alpha_l = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / d
        comp = pred.unsqueeze(1) * fg + (1 - pred.unsqueeze(1)) * bg
        comp_l = ((comp - image).abs().mean(1) * u).sum(dim=(-2, -1)) / d
        return (alpha_l + 0.5 * comp_l).mean()
    return loss_fn
