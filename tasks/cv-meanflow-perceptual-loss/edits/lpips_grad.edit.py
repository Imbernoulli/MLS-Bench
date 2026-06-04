"""Strong baseline: MSE-on-velocity + LPIPS + Sobel grad + multiscale-L1
with quadratic (1-t)^2 perceptual decay.

This is the spatial-domain perceptual recipe. All three auxiliary signals
operate on `x_denoised = x_t - t*v_pred`:
- LPIPS feature-space distance (Zhang 2018)
- Sobel edge-gradient L1 (compute_gradient_loss from perceptual_utils)
- Multi-scale downsampled L1 (compute_multiscale_loss from perceptual_utils)

Additionally adds a small Charbonnier-smoothed velocity loss as an L1-style
robust term alongside MSE — Charbonnier (sqrt(x^2 + eps^2)) is standard
in super-resolution (Lai 2017 LapSRN) and complements MSE with a more
robust pixel error.

Clean linear combination (NO inverse-loss adaptive reweighting — the
previous baseline's `weight = 1/(loss_mse + 1e-3)` poison caused
divergence at 35-40k). Mask t<=0.1 to skip the numerical edge case.
"""

_FILE = "alphaflow-main/custom_train_perceptual.py"

_LPIPS_GRAD = '''\
            # MSE on velocity + Charbonnier smooth-L1 pixel loss on velocity
            err = pred_mean_vel - mean_vel_target
            loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
            loss_charb = torch.sqrt(err ** 2 + 1e-6).flatten(1).mean(1)

            # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
            x_denoised = x_t - t * pred_mean_vel
            t_flat = t.view(B)
            mask = (t_flat > 0.1)
            perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()

            loss_lpips = torch.zeros(B, device=device)
            loss_grad = torch.zeros(B, device=device)
            loss_multi = torch.zeros(B, device=device)
            if mask.any():
                xd = x_denoised[mask].clamp(-1, 1).float()
                xc = x[mask].clamp(-1, 1).float()
                loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
                loss_grad[mask] = compute_gradient_loss(xd, xc).float()
                loss_multi[mask] = compute_multiscale_loss(xd, xc).float()

            loss_total = (
                loss_mse_unscaled
                + 0.1 * loss_charb
                + perceptual_w * (0.5 * loss_lpips + 0.3 * loss_grad + 0.2 * loss_multi)
            )
            loss = loss_total.mean()
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 445,
        "end_line": 462,
        "content": _LPIPS_GRAD,
    },
]
