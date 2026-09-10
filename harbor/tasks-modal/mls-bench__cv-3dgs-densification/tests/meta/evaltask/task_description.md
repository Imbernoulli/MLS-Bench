# 3D Gaussian Splatting Densification Strategy

## Objective

Design a densification strategy for 3D Gaussian Splatting (3DGS) that improves
novel view synthesis quality on real-world scenes under a fixed training and
rendering pipeline.

## Background

3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) represents scenes as
collections of anisotropic 3D Gaussians optimized via differentiable
rasterization. A central component of training is the **densification strategy**,
which controls how Gaussians are added, split, pruned, or otherwise reorganized
during optimization. Common operations include:

- **Clone** small Gaussians in under-reconstructed regions.
- **Split** large Gaussians into smaller ones to recover finer detail.
- **Prune** transparent or oversized Gaussians.
- **Reset** opacities periodically to encourage pruning of redundant Gaussians.

Recent work proposes various refinements:

- **AbsGS** (Ye et al., arXiv:2404.10484) — homodirectional view-space gradient
  using the absolute value of per-pixel sub-gradients to overcome
  over-reconstruction caused by gradient cancellation.
- **Mini-Splatting** (Fang & Wang, arXiv:2403.14166) — blur-aware splitting and
  importance-weighted stochastic sampling for Gaussian count control.
- **3DGS-MCMC** (Kheradmand et al., NeurIPS 2024 Spotlight, arXiv:2404.09591) —
  treats densification as Markov-Chain Monte Carlo sampling, replacing cloning
  with a relocation step that preserves the sampled distribution.
- **Taming-3DGS** (Mallick et al., SIGGRAPH Asia 2024, arXiv:2406.15643) —
  budgeted per-step densification controlled by maximum gradient blending.
- **EDC: Efficient Density Control** (Deng et al., arXiv:2411.10133) — long-axis
  splitting with explicit child-Gaussian opacity control plus recovery-aware
  pruning.

## Implementation Contract

Implement a `CustomStrategy` class in `custom_strategy.py`. The strategy
controls the full lifecycle of Gaussians during training via two hooks called
by the training loop:

```python
@dataclass
class CustomStrategy(Strategy):
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        # Initialize running statistics for the strategy.
        ...

    def step_pre_backward(self, params, optimizers, state, step, info):
        # Called BEFORE loss.backward(). Use to retain gradients.
        ...

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # Called AFTER loss.backward() and optimizer.step().
        # Implement densification / pruning logic here.
        ...
```

### Available Operations (`gsplat.strategy.ops`)

- `duplicate(params, optimizers, state, mask)` — clone selected Gaussians.
- `split(params, optimizers, state, mask)` — split selected Gaussians (sample 2
  new positions from the covariance).
- `remove(params, optimizers, state, mask)` — remove selected Gaussians.
- `reset_opa(params, optimizers, state, value)` — reset all opacities to a
  value.
- `relocate(params, optimizers, state, mask, binoms, min_opacity)` — relocate
  dead Gaussians on top of live ones.
- `sample_add(params, optimizers, state, n, binoms, min_opacity)` — add new
  Gaussians sampled from the opacity distribution.
- `inject_noise_to_position(params, optimizers, state, scaler)` — perturb
  positions with Gaussian noise.

### Available Information

The `info` dict passed in by the rasterizer contains:

- `means2d` — 2D projected means (with `.grad` after backward).
- `width`, `height` — image dimensions.
- `n_cameras` — number of cameras in the batch.
- `radii` — screen-space radii per Gaussian.
- `gaussian_ids` — which Gaussians are visible.

The `params` dict contains:

- `means` — `[N, 3]` positions.
- `scales` — `[N, 3]` log-scales (use `torch.exp(...)` for actual scales).
- `quats` — `[N, 4]` rotation quaternions.
- `opacities` — `[N]` logit-opacities (use `torch.sigmoid(...)` for actual
  opacities).
- `sh0`, `shN` — spherical-harmonic colour coefficients.

### Fixed Pipeline

The following are FIXED across all strategies and must not be changed:

- Renderer: `gsplat` CUDA rasterizer.
- Optimizer: AdamW with per-parameter learning rates.
- Photometric loss: `0.8 * L1 + 0.2 * SSIM` per training step.
- Training: 30,000 steps per scene.
- SH degree: 3 (increased gradually during training).

## Baselines

| Baseline   | Description |
|------------|-------------|
| `absgrad`  | gsplat `DefaultStrategy` with the AbsGS absolute-gradient criterion (Ye et al., arXiv:2404.10484). |
| `taming`   | Taming-3DGS budgeted densification with max-grad blending (Mallick et al., arXiv:2406.15643), combined with the AbsGS gradient and the revised opacity formula. |
| `edc`      | Taming densification combined with EDC long-axis splitting and recovery-aware pruning (Deng et al., arXiv:2411.10133). |

## Evaluation

Evaluation uses Mip-NeRF 360 scenes (Barron et al., 2022) with every 8th image
held out for testing. Metrics:

| Metric  | Direction | Description |
|---------|-----------|-------------|
| **PSNR**  | higher is better | Peak signal-to-noise ratio (primary metric). |
| **SSIM**  | higher is better | Structural similarity. |
| **LPIPS** | lower is better  | Learned perceptual similarity. |

Scoring uses per-scene PSNR. The contribution should be a transferable
densification rule, not a change to the renderer, photometric loss, optimizer,
dataset, or evaluation protocol.
