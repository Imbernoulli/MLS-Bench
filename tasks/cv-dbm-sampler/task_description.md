# Custom Sampler for Diffusion Bridge Models

## Objective

Design a sampling algorithm for Diffusion Bridge Models (DBMs) that improves
conditional generation quality on image-to-image translation tasks under a
strict per-sample budget on the number of denoiser calls.

Implement the algorithm inside the `sample_custom_bridge` function in
`dbim-codebase/ddbm/karras_diffusion.py`. The evaluation pipeline calls this function to
generate target images from source conditions.

## Background

Diffusion Bridge Models construct stochastic or deterministic paths between
two arbitrary distributions (e.g., a sketch and a realistic image), enabling
high-quality image-to-image translation directly without first mapping to
unconditional Gaussian noise. The benchmark provides three reference families:

- **DDBM — Denoising Diffusion Bridge Models** (Zhou et al., 2023,
  arXiv:2309.16948). Simulates the bridge using a continuous Fokker–Planck /
  reverse-SDE formulation. Vanilla samplers require many denoiser calls.
- **DBIM — Diffusion Bridge Implicit Models** (Zheng et al., ICLR 2025,
  arXiv:2405.15885). Generalizes DDBMs via non-Markovian bridges sharing the
  same marginals; analytically decouples each step into closed-form
  coefficients (`coeff_x0_hat`, `coeff_xT`, `coeff_xs`) so the sampler can take
  large jumps and supports interpolation between deterministic and stochastic
  updates.
- **ECSI — Endpoint-Conditioned Stochastic Interpolants** (Tang et al.,
  arXiv:2410.21553, "Exploring the Design Space of Diffusion Bridge Models").
  Uses a `z_hat` (noise) reparameterization with explicit stochasticity control
  (`ε_t = η · (γ γ̇ − (α̇/α) γ²)`) and falls back to a DBIM step on the final
  two timesteps for endpoint sharpness.

The research question is whether a better transition rule can synthesize the
strengths of these families — or introduce a new mathematical update — to
reduce FID under a small NFE budget.

## Implementation Contract

Your novel logic lives inside this function (do not change the signature or
the return tuple):

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    # x: initial bridge state (e.g., source image with noise).
    # ts: time schedule tensor, monotonically decreasing from t_max to 0.
    # eta: stochasticity scale.

    # ... your custom sampling logic ...

    # Must return exactly these 6 values, in this order:
    return x, path, nfe, pred_x0, ts, first_noise
```

**Constraints:**

- Do not modify the function name, arguments, or return structure. The outer
  `sample.py` loop strictly expects
  `(final_image, sampling_path, num_function_evals, predicted_x0_list, time_schedule, initial_noise)`.
- Do not alter how external hyperparameters (e.g. `guidance_scale`,
  `corrupt_scale`) are parsed from environment variables.
- The evaluation pipeline wraps the denoiser with a counter. You may call
  `denoiser(...)` at most `len(ts)` times per sample — the
  `(len(ts) + 1)`-th call raises `RuntimeError: NFE_BUDGET_EXCEEDED` and the
  run is rejected. How you allocate those calls and schedule stochasticity is
  entirely your choice.
- Preserve `mask` semantics for restoration / inpainting workloads.

## Baselines

| Baseline           | Description |
|--------------------|-------------|
| `dbim`             | Diffusion Bridge Implicit Models (Zheng et al., ICLR 2025, arXiv:2405.15885) — fast non-Markovian bridge sampler with closed-form coefficients. |
| `dbim_high_order`  | DBIM with the high-order ODE solver derived in the same paper. |
| `ddbm`             | Reverse-SDE sampler from the original DDBM paper (Zhou et al., arXiv:2309.16948). Used here as a high-NFE reference; its budget is not available to the agent. |
| `ecsi`             | Endpoint-conditioned stochastic interpolant sampler (Tang et al., arXiv:2410.21553). |

## Evaluation

Evaluation runs multiple image-to-image and restoration workloads (e.g.
Edges→Handbags, ImageNet center-inpainting). Metric: **FID — Fréchet Inception
Distance**, lower is better. The parser also verifies the actual number of
denoiser calls per sample and rejects runs that exceed the allowed budget.

The agent-facing budget is **NFE = 5 denoiser calls per sample**. The DDBM
high-NFE baseline is a reference point only — it does not grant additional
function evaluations to your sampler.

The contribution should be the sampler update rule. Keep dataset handling,
external hyperparameter parsing, and evaluation scripts unchanged.

## Implementation Hints

- The marginal distributions (the schedules for `x_0`, `x_T`, and noise) are
  fixed by the underlying VP schedule; do not arbitrarily alter the
  closed-form coefficients. Focus on how to modulate the SDE / ODE balance
  across steps.
- Stochasticity scheduling matters: how should `eta` (or an equivalent
  per-step `ε_t`) vary across the trajectory to balance exploration with
  endpoint sharpness?
