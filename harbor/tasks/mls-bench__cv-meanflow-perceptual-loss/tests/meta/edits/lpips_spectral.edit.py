"""Strongest baseline: full perceptual stack (spatial + frequency).

Combines qwen3-max's winning 3-signal recipe (LPIPS + Sobel grad + multiscale L1)
with an additional FFT-magnitude L1 (Mathieu 2016 / Fuoli 2021) to cover both
spatial and frequency domains simultaneously.

Components:
- MSE on velocity (base)
- LPIPS feature-space distance
- Sobel-gradient L1 (spatial edge)
- Multi-scale downsampled L1 (multi-resolution)
- FFT magnitude L1 (global frequency spectrum)

All auxiliary terms decayed by (1-t)^2 so they concentrate on low-noise
samples where x_denoised is meaningful. Mask t<=0.1 for numerical stability.
Clean linear combination (no inverse-loss adaptive reweighting).
"""

_FILE = "alphaflow-main/custom_train_perceptual.py"

_LPIPS_SPECTRAL = '''\
            # MSE on velocity
            err = pred_mean_vel - mean_vel_target
            loss_mse_unscaled = (err ** 2).flatten(1).mean(1)

            # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
            x_denoised = x_t - t * pred_mean_vel
            t_flat = t.view(B)
            mask = (t_flat > 0.1)
            perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()

            loss_lpips = torch.zeros(B, device=device)
            loss_grad = torch.zeros(B, device=device)
            loss_multi = torch.zeros(B, device=device)
            loss_spec = torch.zeros(B, device=device)
            if mask.any():
                xd = x_denoised[mask].clamp(-1, 1).float()
                xc = x[mask].clamp(-1, 1).float()
                loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
                loss_grad[mask] = compute_gradient_loss(xd, xc).float()
                loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
                # FFT magnitude L1: per-channel rfft2, abs, L1 of difference
                fd = torch.fft.rfft2(xd, dim=(-2, -1)).abs()
                fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
                loss_spec[mask] = (fd - fc).abs().mean(dim=(1, 2, 3)).float()

            loss_total = (
                loss_mse_unscaled
                + perceptual_w * (
                    0.5 * loss_lpips
                    + 0.3 * loss_grad
                    + 0.2 * loss_multi
                    + 0.2 * loss_spec
                )
            )
            loss = loss_total.mean()
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 384,
        "end_line": 401,
        "content": _LPIPS_SPECTRAL,
    },
]
