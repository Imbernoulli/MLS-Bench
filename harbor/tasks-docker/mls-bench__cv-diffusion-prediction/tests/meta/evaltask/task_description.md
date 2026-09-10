# Diffusion Prediction Parameterization

## Objective

Design a prediction parameterization for unconditional CIFAR-10 diffusion that
improves FID under a fixed UNet backbone, training procedure, and DDIM sampler.

## Background

In DDPM training (Ho et al., 2020, arXiv:2006.11239), the model is shown a
noisy sample

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
```

and trained to predict a chosen target. Three standard parameterizations:

1. **Epsilon prediction** (Ho et al., 2020, arXiv:2006.11239) — predict the
   noise `epsilon`. Standard DDPM choice.
2. **`x_0` prediction** — directly predict the clean image `x_0`.
3. **`v` prediction** (Salimans & Ho, ICLR 2022, arXiv:2202.00512,
   "Progressive Distillation for Fast Sampling of Diffusion Models") —
   predict the velocity `v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x_0`.

The three are mathematically interchangeable (any one can be converted to the
others), but they give different loss landscapes, signal scaling across
timesteps, and gradient magnitudes, leading to different FID under a finite
training budget.

## Implementation Contract

You are given `custom_train.py`, a self-contained training script that trains
a UNet (`google/ddpm-cifar10-32` style architecture) on CIFAR-10. The
editable region contains two coupled functions:

1. `compute_training_target(x_0, noise, timesteps, schedule)` — defines what
   the model should predict during training.
2. `predict_x0(model_output, x_t, timesteps, schedule)` — recovers the
   predicted clean image from the model's output. Used during DDIM sampling.

These two functions must be **consistent**: the sampling procedure must
correctly invert the training parameterization.

The `schedule` dict provides precomputed noise-schedule tensors:

- `alphas_cumprod` — cumulative product of `(1 - beta)`.
- `sqrt_alpha` — `sqrt(alphas_cumprod)`.
- `sqrt_one_minus_alpha` — `sqrt(1 - alphas_cumprod)`.

## Fixed Pipeline

The following are fixed across baselines and submissions:

- Dataset: CIFAR-10 (32×32, unconditional).
- Backbone: `UNet2DModel` (diffusers) at three channel scales:
  - Small:  `block_out_channels=(64, 128, 128, 128)`, ~9M params, batch 128.
  - Medium: `block_out_channels=(128, 256, 256, 256)`, ~36M params, batch 128.
  - Large:  `block_out_channels=(160, 320, 320, 320)`, ~55M params, batch 128.
- Training: 35,000 steps per scale, AdamW lr=2e-4, EMA rate 0.9995,
  multi-GPU DDP.
- Inference: 50-step DDIM (Song et al., 2020, arXiv:2010.02502).
- Metric: FID computed by clean-fid against the CIFAR-10 train set
  (50,000 samples), lower is better.

## Baselines

| Baseline  | Description |
|-----------|-------------|
| `epsilon` | Predict `epsilon` (Ho et al., 2020, arXiv:2006.11239). DDPM default. |
| `x0pred`  | Predict the clean image `x_0` directly. |
| `vpred`   | Predict the velocity `v = sqrt(alpha) * epsilon - sqrt(1 - alpha) * x_0` (Salimans & Ho, ICLR 2022, arXiv:2202.00512). |

## Evaluation

Evaluation trains the candidate parameterization at the channel scales above
and scores with clean-fid against CIFAR-10; lower FID is better. The
contribution should be a transferable target parameterization, not a change
to architecture, dataset, optimizer, noise schedule, sampling procedure, or
metric computation.
