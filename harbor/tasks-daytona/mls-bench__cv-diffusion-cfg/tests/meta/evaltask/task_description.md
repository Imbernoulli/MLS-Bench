# Diffusion Model: Classifier-Free Guidance Optimization

## Objective

Design a classifier-free guidance (CFG) method for text-to-image diffusion
that improves generation quality across Stable Diffusion model variants under
a fixed sampling pipeline.

## Background

Classifier-free guidance (Ho & Salimans, 2022, arXiv:2207.12598) combines
unconditional and conditional noise predictions to trade off prompt alignment
and image quality. The standard formula is:

```
noise_pred = noise_uc + cfg_scale * (noise_c - noise_uc)
```

where `noise_uc` is the unconditional noise prediction, `noise_c` is the
text-conditioned noise prediction, and `cfg_scale` is typically in the range
7.5 – 12.5 for high-quality samples.

Standard CFG has well-documented limitations: it can cause mode collapse, over-
saturated colours, off-manifold sampling trajectories that hurt invertibility,
and a sensitive dependence on guidance scale. Recent work proposes manifold-
constrained alternatives:

- **CFG++** (Chung et al., ICLR 2025, arXiv:2406.08070) — reformulates CFG as
  decomposed reverse diffusion sampling: instead of renoising with the
  guided prediction, renoise with the unconditional prediction, keeping the
  latent on the data manifold and enabling small guidance scales (0 < λ < 1).
- **Zero-init / skip-step variants** — skip the first few sampling steps
  before applying guidance to reduce trajectory error at the highest noise
  levels.

## Implementation Contract

Implement the guidance rule for both Stable Diffusion v1.5 and SDXL by editing
the marked editable regions of two files:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
   (`sample()` method). Available helpers:
   `self.get_text_embed()`, `self.initialize_latent()`,
   `self.predict_noise()`, `self.alpha(t)`.
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL
   (`reverse_process()` method). Available helpers:
   `self.initialize_latent(size=...)`, `self.predict_noise()`,
   `self.scheduler.alphas_cumprod[t]`.

The contribution may change how conditional and unconditional predictions are
combined, how the latent is renoised, or how guidance strength varies with
time, but it should not change the prompt set, model weights, the number of
allowed denoiser evaluations, or evaluation code.

## Baselines

| Baseline   | Description |
|------------|-------------|
| `cfg`      | Standard classifier-free guidance (Ho & Salimans, arXiv:2207.12598): renoise with the guided noise prediction. |
| `cfgpp`    | CFG++ (Chung et al., ICLR 2025, arXiv:2406.08070): renoise with the unconditional noise prediction, keeping the trajectory on the data manifold. |
| `zeroinit` | Zero-init + rescaled standard CFG: skip guidance for the first K = 2 sampling steps, then renoise with the guided prediction. |

## Fixed Pipeline

- Models: Stable Diffusion v1.5 and SDXL (frozen weights).
- Sampling: fixed sampler call structure with a fixed step budget.
- Prompts: shared evaluation prompt set across all baselines.

## Evaluation

Evaluation runs the text-to-image sampling pipeline on the model variants
above. The task-visible metric and official score use **FID** computed against
a reference image set (lower is better). The generation script may compute
CLIP diagnostics internally, but CLIP is not part of the task score or
agent-visible feedback.

A good method should improve image quality without sacrificing the
prompt-following behaviour provided by guidance.
