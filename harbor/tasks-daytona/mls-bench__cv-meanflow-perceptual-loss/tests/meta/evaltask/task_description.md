# Flow Matching with Perceptual Loss

## Objective

Design an auxiliary training loss for CIFAR-10 flow matching that improves
sample FID under a fixed DiT backbone, MeanFlow training objective, and
ten-step Euler sampler.

## Background

Flow matching trains a network to predict velocity fields that transport
samples from noise to data. **MeanFlow** (Geng et al., 2025,
arXiv:2505.13447, "Mean Flows for One-step Generative Modeling") is a
flow-matching variant that learns the *average* velocity field over a time
interval and supports very-low-step (down to single-step) generation. The
canonical training loss is mean squared error on the predicted mean velocity:

```
loss_mse = || v_pred - v_target ||^2
```

However, the predicted velocity also implies a denoised image at every step:

```
x_denoised = x_t - t * v_pred
```

so auxiliary losses can be applied on `x_denoised` (image-space, perceptual,
gradient, multiscale, frequency-domain) to encourage the network to produce
high-quality images, not only accurate velocities.

## Implementation Contract

You are given `custom_train_perceptual.py`, a self-contained training script
that trains a small DiT (Peebles & Xie, ICCV 2023, arXiv:2212.09748) on
CIFAR-10 using MeanFlow. The editable region is the loss computation in the
training loop, e.g.:

```python
# Current: pure MSE on velocity.
loss_mse = ((pred_mean_vel - mean_vel_target) ** 2).mean()
loss = loss_mse
```

The fixed code already exposes helpers you may use to build perceptual /
auxiliary losses on `x_denoised`:

- `lpips_fn(x_denoised, x_target)` — LPIPS perceptual loss.
- `compute_gradient_loss(x_denoised, x_target)` — Sobel-style gradient-domain
  loss.
- `compute_multiscale_loss(x_denoised, x_target)` — multi-resolution loss.

**Stability constraint:** apply auxiliary losses only when `t > 0.1`. At very
small `t` the implied `x_denoised` becomes ill-conditioned and auxiliary
gradients dominate the velocity target.

## Fixed Pipeline

- Dataset: CIFAR-10 (32×32).
- Models: DiT backbones at the configured scales:
  - Small:  hidden_size=512, depth=8, num_heads=8.
  - Medium: hidden_size=640, depth=10, num_heads=10.
  - Large:  hidden_size=768, depth=12, num_heads=12.
- Training: 40,000 steps, batch size 256.
- Inference: 5-step Euler sampler.
- Metric: FID computed by clean-fid against the CIFAR-10 train set, lower is
  better.

## Baselines

| Baseline         | Description |
|------------------|-------------|
| `mse_base`       | Pure MSE on velocity — clean linear formulation, the floor reference. |
| `lpips_grad`     | MSE + Charbonnier-smoothed L1 on velocity + LPIPS + Sobel gradient + multiscale L1 on `x_denoised`, with a `(1 − t)^2` perceptual schedule and a `t ≤ 0.1` mask (spatial-domain perceptual recipe). |
| `lpips_spectral` | `lpips_grad` stack augmented with an FFT-magnitude L1 term on `x_denoised` (spatial + frequency-domain recipe). |

## Evaluation

Evaluation trains on CIFAR-10 at the configured scales / budgets and samples
with the fixed ten-step Euler sampler. Scoring uses FID per scale; lower is
better.

A useful method should improve visual sample quality without destabilizing the
velocity target. Auxiliary losses must be applied only where `x_denoised` is
numerically meaningful. Do not change the architecture, data pipeline,
sampler, number of evaluation steps, or metric computation.
