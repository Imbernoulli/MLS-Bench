"""GOOD loss probe: unknown-band-focused alpha-L1 (+ composition).

Restricts the alpha-L1 to the UNKNOWN band (where matting is actually solved) and
adds a composition term, so the optimiser spends its capacity on the hard transition
-> lower SAD. This is the matting-standard practice (loss on the unknown region).
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
