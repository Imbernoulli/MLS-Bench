"""Good baseline for cv-matting-loss-design: alpha-L1 + composition + Laplacian.

Deep Image Matting (Xu et al. 2017) combines an alpha-prediction loss with a
COMPOSITION loss (equal weight, w=0.5): penalise |I - (a*F + (1-a)*B)|, tying the
matte to the observed image. A LAPLACIAN-PYRAMID loss (Context-Aware Matting, Hou &
Liu 2019) adds multi-scale structure that sharpens the soft transition. Together
they lower SAD / MSE / gradient error in the unknown band with clear headroom over
plain alpha-L1. Reference: vendor/image-matting/baselines/loss_comp_lap.py
"""


def get_matting_loss():
    import torch
    import torch.nn.functional as F

    def _lap_pyr(x, levels=4):
        # x: (B,H,W) -> list of laplacian levels
        x = x.unsqueeze(1)
        ker = torch.tensor([1., 4., 6., 4., 1.], device=x.device)
        ker = (ker[:, None] * ker[None, :])
        ker = (ker / ker.sum()).view(1, 1, 5, 5)
        pyr = []
        cur = x
        for _ in range(levels):
            blur = F.conv2d(F.pad(cur, (2, 2, 2, 2), mode="reflect"), ker)
            down = blur[:, :, ::2, ::2]
            up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)
        return pyr

    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        u = unknown.float()
        d = u.sum(dim=(-2, -1)).clamp(min=1.0)
        alpha_l = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / d
        comp = pred.unsqueeze(1) * fg + (1 - pred.unsqueeze(1)) * bg
        comp_l = ((comp - image).abs().mean(1) * u).sum(dim=(-2, -1)) / d
        lp, lg = _lap_pyr(pred * u), _lap_pyr(gt * u)
        lap_l = sum((2 ** i) * (a - b).abs().mean() for i, (a, b) in enumerate(zip(lp, lg)))
        return (alpha_l + 0.5 * comp_l).mean() + 0.1 * lap_l
    return loss_fn
