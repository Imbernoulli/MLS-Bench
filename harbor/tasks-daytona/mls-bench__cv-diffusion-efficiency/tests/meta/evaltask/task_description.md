# Diffusion Model: Sampler Efficiency Optimization

## Objective

Design a sampling algorithm for text-to-image diffusion models that achieves
high generation quality with a fixed budget of NFE = 50 denoiser evaluations.

## Background

Diffusion models generate images by iteratively denoising from random noise.
Different samplers differ in how they update the latent after each model
prediction. The general structure of one step is:

```python
for step, t in enumerate(timesteps):
    # 1. Predict noise.
    noise_pred = model(zt, t, text_embedding)
    # 2. Estimate clean image (Tweedie's formula).
    z0t = (zt - sigma_t * noise_pred) / alpha_t
    # 3. Update to next step (this differs across samplers).
    zt_next = update_rule(zt, z0t, noise_pred, t, t_next)
```

Reference families:

- **DDIM** (Song et al., ICLR 2021, arXiv:2010.02502) — first-order ODE
  solver, deterministic, simple update rule.
- **DPM-Solver++** (Lu et al., 2022, arXiv:2211.01095) — high-order solvers
  for the diffusion ODE in data-prediction form.
  - **DPM-Solver++(2M)** — second-order multistep variant, reuses the
    previous denoiser output.
  - **DPM-Solver++(2S)** — second-order singlestep variant, smaller
    high-order error constant.
  - **DPM-Solver++(3M) SDE** — third-order multistep stochastic variant for
    guided sampling.

A useful method may use time-dependent coefficients, history (multistep),
predictor-corrector structure, or guidance-aware renoising — but it must
respect the fixed function-evaluation budget.

## Implementation Contract

Implement the update rule for both Stable Diffusion v1.5 and SDXL by editing
the marked editable regions of two files:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
   (`sample()` method). Available helpers:
   `self.get_text_embed()`, `self.initialize_latent()`,
   `self.predict_noise()`, `self.alpha(t)`.
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL
   (`reverse_process()` method). Available helpers:
   `self.initialize_latent(size=...)`, `self.predict_noise()`,
   `self.scheduler.alphas_cumprod[t]`.

The contribution must respect a fixed budget of **NFE = 50** denoiser calls
per sample.

The budget is measured, not trusted. The evaluation script replaces the
solver's `self.unet` with a counting wrapper, so every UNet forward the sampler
makes, through `self.predict_noise()` or `self.unet(...)` directly, is counted
for each image. One NFE is one UNet forward on the image's latent: the batched
unconditional + conditional pair of classifier-free guidance counts as 1, and
so does a single-branch call. A forward over more than two latent rows counts
ceil(rows / 2). A run that spends more than 50 NFE on any image is rejected and
records no FID; spending fewer is allowed. All denoiser evaluations must go
through `self.unet` / `self.predict_noise()`, and a solver that keeps another
handle on the UNet is rejected.

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `ddim`      | DDIM (Song et al., ICLR 2021, arXiv:2010.02502). First-order deterministic. |
| `dpm3m_sde` | DPM-Solver++(3M) SDE multistep variant (Lu et al., 2022, arXiv:2211.01095). |
| `dpm2s`     | DPM-Solver++(2S) second-order singlestep variant (same paper). Two evaluations per step, so 25 steps. |

## Fixed Pipeline

- Models: Stable Diffusion v1.5 and SDXL (frozen weights).
- Prompt set: shared evaluation prompts across all baselines.
- NFE budget: 50 denoiser calls per sample.

## Evaluation

Evaluation runs text-to-image sampling on the model variants above. The
task-visible metric and official score use **FID** computed against a reference
image set (lower is better). The generation script may compute CLIP diagnostics
internally, but CLIP is not part of the task score or agent-visible feedback.

The method should improve image quality across variants without changing
prompts, model weights, allowed function-evaluation budget, or metric
computation.
