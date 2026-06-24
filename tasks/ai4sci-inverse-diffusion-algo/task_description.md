# Task: Inverse Problem Algorithm Design with Diffusion Priors

## Research Question
Design a novel algorithm for solving scientific inverse problems using pre-trained diffusion model priors. Given a forward operator A and observation `y = A(x) + noise`, the algorithm should reconstruct `x` by leveraging a learned diffusion prior `p(x)`.

## Background
Diffusion models learn rich priors `p(x)` over signal distributions. For inverse problems, we want to sample from the posterior `p(x|y) ∝ p(y|x) p(x)`. Existing approaches include:

- **DPS — Diffusion Posterior Sampling** (Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023; arXiv:2209.14687). Uses the score `∇_x log p(x)` from the diffusion model and adds measurement guidance `∇_x log p(y|x)` at each denoising step. Code: https://github.com/DPS2022/diffusion-posterior-sampling.
- **REDDiff — Variational / Regularization-by-Denoising-Diffusion** (Mardani, Song, Kautz, Vahdat, "A Variational Perspective on Solving Inverse Problems with Diffusion Models", ICLR 2024; arXiv:2305.04391). Variational formulation that yields a regularization-by-denoising update where denoisers at different timesteps concurrently impose structural constraints. Code: https://github.com/NVlabs/RED-diff.
- **LGD — Loss-Guided Diffusion** (Song et al., "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation", ICML 2023). Estimates the guidance term via Monte Carlo sampling around the denoised estimate to reduce bias of point-estimate approximations.

## What to Implement
Implement the `Custom` class in `InverseBench/algo/custom.py`. You must implement:
1. `__init__`: Set up your algorithm (schedulers, optimizers, hyperparameters).
2. `inference(observation, num_samples)`: Given observation `y`, return reconstructed `x`.

## Available Components
- `self.net(x, sigma)` → denoised estimate (Tweedie's formula: E[x_0 | x_t]).
- `self.forward_op.forward(x)` → compute `A(x)`.
- `self.forward_op.gradient(x, y, return_loss=True)` → `(∇_x ||A(x) - y||², loss)`.
- `self.forward_op.loss(x, y)` → `||A(x) - y||²`.
- `Scheduler(num_steps, schedule, timestep, scaling)` → diffusion noise schedule.
- `DiffusionSampler(scheduler).sample(model, x_start)` → unconditional sampling.

The pretrained denoiser, the forward-operator definitions, and the evaluation problems are fixed; the algorithm only chooses how to combine these pieces.

## Evaluation
The algorithm is tested on three scientific inverse problems:
1. **Inverse Scattering** (optical tomography): Recover permittivity from scattered EM fields. Metrics: PSNR, SSIM.
2. **Black Hole Imaging** (radio astronomy): Reconstruct black hole images from sparse interferometric observations (EHT data). Metrics: PSNR, blur-PSNR (f=15), closure-phase chi-squared.
3. **FFHQ256 Image Inpainting** (computer vision): Recover an FFHQ-256 face image from a masked observation (box mask) with additive Gaussian noise (σ=0.05). The forward operator is a fixed pixel-wise mask. Metrics: PSNR, SSIM, LPIPS.

Higher PSNR/SSIM is better; lower LPIPS and chi-squared are better.

## Editable Region
The entire `InverseBench/algo/custom.py` file is editable. You may define any helper classes/functions within this file.
