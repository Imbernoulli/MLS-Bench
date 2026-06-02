# MLS-Bench: cv-3dgs-regularizer

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gsplat/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `gsplat/custom_regularizer.py`
- editable lines **37–51**




## Readable Context


### `gsplat/custom_regularizer.py`  [EDITABLE — lines 37–51 only]

```python
     1: """Custom regularizer for 3D Gaussian Splatting training.
     2: 
     3: Your `compute_regularizer` is added directly to the photometric loss
     4: (0.8 * L1 + 0.2 * SSIM) every training step. The goal is to design a
     5: regularization term, computed only from the Gaussian parameters (no
     6: ground-truth depth or normals), that improves novel-view reconstruction
     7: quality on MipNeRF360 scenes — higher PSNR / SSIM, lower LPIPS.
     8: 
     9: Call signature:
    10: 
    11:     compute_regularizer(splats, step, scene_scale) -> scalar tensor
    12: 
    13: Where:
    14:     splats        torch.nn.ParameterDict, keys below (first dim = N Gaussians)
    15:     step          int, current iteration (0 .. max_steps-1)
    16:     scene_scale   float, approx scene radius for distance normalization
    17: 
    18: splats keys (tensor shapes):
    19:     "means"       [N, 3]      world-space Gaussian centers
    20:     "scales"      [N, 3]      log-scales; torch.exp(...) for actual scale
    21:     "quats"       [N, 4]      rotation quaternion (unnormalized)
    22:     "opacities"   [N]         logit-opacities; torch.sigmoid(...) for [0,1]
    23:     "sh0"         [N, 1, 3]   DC spherical-harmonic coefficients
    24:     "shN"         [N, K, 3]   higher-order SH, K=(max_sh_deg+1)**2 - 1
    25: 
    26: Evaluation: PSNR (higher is better), SSIM (higher is better),
    27:             LPIPS (lower is better). Measured on held-out views.
    28: """
    29: 
    30: import torch
    31: import torch.nn.functional as F
    32: 
    33: 
    34: # ============================================================================
    35: # EDITABLE REGION — implement your regularizer in the block below.
    36: # ============================================================================
    37: 
    38: def compute_regularizer(splats, step, scene_scale):
    39:     """Return a scalar tensor added to the photometric loss.
    40: 
    41:     TODO: design a regularizer that improves reconstruction quality.
    42:     Hints:
    43:       - Parameter-level penalties (scale, opacity, SH) are cheap and often
    44:         effective. Tune the weights per scene scale.
    45:       - Neighbor-based priors (e.g. kNN over splats["means"]) add a small
    46:         amount of spatial structure.
    47:       - Keep the cost bounded: this function is called every training step.
    48:     """
    49:     # Default: no regularization.
    50:     return torch.zeros((), device=splats["means"].device)
    51: 
    52: # ============================================================================
    53: # End editable region. The training loop imports compute_regularizer from
    54: # this file and adds its return value to the photometric loss unchanged.
    55: # ============================================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `none` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_regularizer.py`:

```python
Lines 37–40:
    34: # ============================================================================
    35: # EDITABLE REGION — implement your regularizer in the block below.
    36: # ============================================================================
    37: 
    38: def compute_regularizer(splats, step, scene_scale):
    39:     """No regularization — zero added to the photometric loss."""
    40:     return torch.zeros((), device=splats["means"].device)
    41: # ============================================================================
    42: # End editable region. The training loop imports compute_regularizer from
    43: # this file and adds its return value to the photometric loss unchanged.
```

### `scale_opa` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_regularizer.py`:

```python
Lines 37–45:
    34: # ============================================================================
    35: # EDITABLE REGION — implement your regularizer in the block below.
    36: # ============================================================================
    37: 
    38: SCALE_REG = 1e-2
    39: OPACITY_REG = 1e-2
    40: 
    41: def compute_regularizer(splats, step, scene_scale):
    42:     """L1 penalty on per-Gaussian scale and opacity."""
    43:     scale_loss = torch.abs(torch.exp(splats["scales"])).mean()
    44:     opa_loss = torch.abs(torch.sigmoid(splats["opacities"])).mean()
    45:     return SCALE_REG * scale_loss + OPACITY_REG * opa_loss
    46: # ============================================================================
    47: # End editable region. The training loop imports compute_regularizer from
    48: # this file and adds its return value to the photometric loss unchanged.
```

### `erank_opa` baseline — editable region  [READ-ONLY — reference implementation]

In `gsplat/custom_regularizer.py`:

```python
Lines 37–61:
    34: # ============================================================================
    35: # EDITABLE REGION — implement your regularizer in the block below.
    36: # ============================================================================
    37: 
    38: # scale_opa (full strength) + erank log-barrier (warmup at step 7000).
    39: SCALE_REG = 1e-2
    40: OPACITY_REG = 1e-2
    41: ERANK_REG = 1e-2
    42: ERANK_WARMUP = 7000
    43: ERANK_EPS = 1e-5
    44: 
    45: def compute_regularizer(splats, step, scene_scale):
    46:     """Compactness L1 (always on) + erank log-barrier (after warmup)."""
    47:     s = torch.exp(splats["scales"])                                # [N, 3]
    48:     a = torch.sigmoid(splats["opacities"])                         # [N]
    49: 
    50:     loss = SCALE_REG * s.mean() + OPACITY_REG * a.mean()
    51: 
    52:     if step >= ERANK_WARMUP:
    53:         s_sq = s * s
    54:         q = s_sq / (s_sq.sum(dim=-1, keepdim=True) + 1e-12)
    55:         H = -(q * (q + 1e-12).log()).sum(dim=-1)
    56:         erank = H.exp()
    57:         barrier = torch.clamp(-torch.log(erank - 1.0 + ERANK_EPS), min=0.0)
    58:         s_min = s.min(dim=-1).values
    59:         loss = loss + ERANK_REG * (barrier.mean() + s_min.mean())
    60: 
    61:     return loss
    62: # ============================================================================
    63: # End editable region. The training loop imports compute_regularizer from
    64: # this file and adds its return value to the photometric loss unchanged.
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
