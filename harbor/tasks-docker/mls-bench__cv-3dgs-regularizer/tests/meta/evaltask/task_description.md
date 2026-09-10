# 3D Gaussian Splatting Regularizer

## Objective

Design a scalar regularizer on 3D Gaussian parameters that improves novel-view
reconstruction quality (higher PSNR / SSIM, lower LPIPS) on Mip-NeRF 360
scenes, without using any depth, normal, or feature-level supervision.

## Background

3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) optimizes millions of
anisotropic Gaussians (means, scales, quaternions, opacities, spherical-harmonic
colours) by gradient descent on a per-scene photometric loss:

```
loss_photo = 0.8 * L1(rendered, gt) + 0.2 * (1 - SSIM(rendered, gt))
```

The photometric loss alone is under-constrained: optimization often produces
elongated "needle" Gaussians, semi-transparent floaters, and other artefacts
that look correct on training views but hurt held-out novel-view quality.
Hand-designed regularizers attack different failure modes:

- **Scale / opacity L1 penalty** (used as a default in 3DGS-MCMC, Kheradmand
  et al., NeurIPS 2024 Spotlight, arXiv:2404.09591) — encourages compact and
  sparse Gaussians.
- **Effective rank regularization** (Hyung et al., NeurIPS 2024,
  arXiv:2406.11672) — penalizes degenerate near-rank-1 needle shapes by
  pushing the effective rank of the covariance toward 2 or higher.
- **Anisotropy / aspect-ratio penalties** — bound `max(scale) / min(scale)` to
  keep Gaussians close to isotropic.
- **Neighbour consistency / blob-prior penalties** — encourage parameter
  smoothness among spatially adjacent Gaussians.

Each is a small, modular addition to the loss, yet can change PSNR by tenths
to ones of a dB on standard benchmarks.

## Implementation Contract

Implement `compute_regularizer(splats, step, scene_scale)` in
`gsplat/custom_regularizer.py`. The scalar return value is added directly to
the photometric loss at every training step, for the entire 30k-step per-scene
optimization.

You may add helpers and module-level constants inside the editable region and
import additional modules. You **must** keep the public signature
`compute_regularizer(splats, step, scene_scale) -> torch.Tensor` returning a
scalar tensor.

### Inputs

- `splats` — `torch.nn.ParameterDict` (first dim is `N` Gaussians):

  | key         | shape       | notes |
  |-------------|-------------|-------|
  | `means`     | `[N, 3]`    | world-space positions |
  | `scales`    | `[N, 3]`    | log-scales; `torch.exp(...)` for actual |
  | `quats`     | `[N, 4]`    | rotation quaternion (unnormalized) |
  | `opacities` | `[N]`       | logit; `torch.sigmoid(...)` for [0, 1] |
  | `sh0`       | `[N, 1, 3]` | DC spherical-harmonic coefficients |
  | `shN`       | `[N, K, 3]` | higher-order SH, K depends on degree |

- `step` — current training iteration (`0` to `max_steps - 1`).
- `scene_scale` — approximate scene radius for distance normalization.

### Output

A scalar `torch.Tensor` (any device). It is added directly to the photometric
loss with no extra scaling, so the regularizer should pre-multiply its own
weights.

## Fixed Pipeline

These are FIXED across baselines and submissions:

- Renderer: `gsplat` CUDA rasterizer.
- Optimizer: AdamW with per-parameter learning rates.
- Photometric loss: `0.8 * L1 + 0.2 * (1 - SSIM)`.
- Densification strategy: gsplat `DefaultStrategy` (original 3DGS
  clone / split / prune).
- Training: 30,000 steps per scene; SH degree 3 (gradually increased).

The regularizer is the only quantity you change.

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `none`      | Returns 0 — photometric loss only. |
| `scale_opa` | L1 on `exp(scales)` and `sigmoid(opacities)` (coefficient 1e-2 each), the default compactness regularizer in 3DGS-MCMC (Kheradmand et al., NeurIPS 2024 Spotlight, arXiv:2404.09591). |
| `erank_opa` | `scale_opa` plus the effective-rank log-barrier regularizer of Hyung et al. (NeurIPS 2024, arXiv:2406.11672) with warmup at step 7000. Pushes the effective rank of each Gaussian toward 2 (planar) while keeping compactness pressure. |

## Evaluation

Evaluation runs on Mip-NeRF 360 scenes (Barron et al., 2022) with every 8th
image held out for testing. Each scene is trained for 30k steps under the
fixed schedule and evaluated on held-out views.

| Metric  | Direction | Description |
|---------|-----------|-------------|
| **PSNR**  | higher is better | Peak signal-to-noise ratio (primary metric). |
| **SSIM**  | higher is better | Structural similarity. |
| **LPIPS** | lower is better  | Learned perceptual similarity. |

## Implementation Hints

- Photometric loss magnitudes are typically `0.03–0.1`; keep the regularizer
  in the `1e-4` to `1e-1` range to avoid overwhelming the data term.
- `step` lets you schedule the regularizer (warmup, cooldown, switch-over).
- `scene_scale` normalizes distances; using `means / scene_scale` gives unit
  coordinates that transfer across scenes.
- Backward flows through every operation. Avoid `log(0)`, `exp(big_number)`,
  divide-by-zero, and other sources of NaN gradients.
- Each scene runs for ~30k iterations. Keep the regularizer at most O(N) in
  the number of Gaussians (no all-pairs `N × N` computations on `means`).
