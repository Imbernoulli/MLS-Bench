"""WEAK loss probe: uniform whole-image alpha-L1 (unknown band NOT emphasised).

Averages the alpha-L1 over EVERY pixel (fg + bg + unknown). The trivial solid
regions (alpha exactly 0 or 1, which any net nails immediately) dominate the mean,
so the gradient signal on the hard UNKNOWN band is diluted -> the transition is
under-fit -> higher SAD in the unknown band.
"""
def get_matting_loss():
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        return (pred - gt).abs().mean()
    return loss_fn
