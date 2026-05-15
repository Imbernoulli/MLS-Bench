# MLS-Bench: cv-dbm-sampler

# Custom Sampler for Diffusion Bridge Models

## Objective

Design a sampling algorithm for Diffusion Bridge Models (DBMs) that improves
conditional generation quality on image-to-image translation tasks under a
strict per-sample budget on the number of denoiser calls.

Implement the algorithm inside the `sample_custom_bridge` function in
`ddbm/karras_diffusion.py`. The evaluation pipeline calls this function to
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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/dbim-codebase/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `dbim-codebase/ddbm/karras_diffusion.py`
- editable lines **459–470**




## Readable Context


### `dbim-codebase/ddbm/karras_diffusion.py`  [EDITABLE — lines 459–470 only]

```python
Lines 448-470:
   448: @torch.no_grad()
   449: def sample_dbim(
   450:     denoiser,
   451:     diffusion,
   452:     x,
   453:     ts,
   454:     eta=1.0,
   455:     mask=None,
   456:     seed=None,
   457:     **kwargs,
   458: ):
   459:     # =================================================================================
   460:     # 🚨 CRITICAL CONSTRAINTS - DO NOT IGNORE! 🚨
   461:     # 1. Function Signature: You must NOT modify the function name, arguments, or return structure.
   462:     # 2. NFE Match (FATAL I/O ERROR): The framework uses the final returned `nfe` to locate
   463:     #    generated files (e.g., expecting `samples_..._nfe5.npz`). You MUST return
   464:     #    `nfe = len(ts) - 1` regardless of the internal call count.
   465:     # =================================================================================
   466:     
   467:     # TODO: Implement your novel sampling kernel here.
   468:     # Ensure the return structure is: return x, path, nfe, pred_x0, ts, first_noise
   469:     
   470:     raise NotImplementedError("Custom sampler not implemented yet.")
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **edges2handbags** — wall-clock budget `4:00:00`, compute share `4.0`
- **Imagenet** — wall-clock budget `4:00:00`, compute share `4.0`
- **DIODE** — wall-clock budget `4:00:00`, compute share `4.0`
- **DIODE_50nfe** — wall-clock budget `4:00:00`, compute share `4.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `dbim` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 459–470:
   456:     seed=None,
   457:     **kwargs,
   458: ):
   459:     x_T = x
   460:     path = []
   461:     pred_x0 = []
   462: 
   463:     ones = x.new_ones([x.shape[0]])
   464:     indices = range(len(ts) - 1)
   465:     indices = tqdm(indices, disable=(dist.get_rank() != 0))
   466: 
   467:     nfe = 0
   468:     x0_hat = denoiser(x, diffusion.t_max * ones)
   469:     generator = BatchedSeedGenerator(seed)
   470:     noise = generator.randn_like(x0_hat)
   471:     first_noise = noise
   472:     if mask is not None:
   473:         x0_hat = x0_hat * mask + x_T * (1 - mask)
```

### `dbim_high_order` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 459–470:
   456:     lower_order_final=True,
   457:     seed=None,
   458:     **kwargs,
   459: ):
   460:     if order not in [2, 3]:
   461:         order=2
   462:     x_T = x
   463:     path = []
   464:     pred_x0 = []
   465: 
   466:     ones = x.new_ones([x.shape[0]])
   467:     indices = range(len(ts) - 1)
   468:     indices = tqdm(indices, disable=(dist.get_rank() != 0))
   469: 
   470:     nfe = 0
   471:     x0_hat = denoiser(x, diffusion.t_max * ones)
   472:     generator = BatchedSeedGenerator(seed)
   473:     noise = generator.randn_like(x0_hat)
```

### `ddbm` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 459–470:
   456: ):
   457:     x_T = x
   458:     path = []
   459:     pred_x0 = []
   460: 
   461:     # DDBM reference baseline: 50-NFE gold-standard reference.
   462:     # Each iteration costs:
   463:     #   * churn euler step (stochastic): 1 denoiser call
   464:     #   * Heun 2nd-order step: 2 denoiser calls (or 1 if ts[i+1]==0)
   465:     # With churn_step_ratio>0: 16 Heun-iters (3 NFE ea.) + 1 final Euler-iter
   466:     # (1 churn + 1 Euler = 2 NFE) = 48 + 2 = 50 NFE total.
   467:     # Terminal ts=0 so the last iteration takes the Euler branch.
   468:     #
   469:     # Agent baselines stay at NFE=5 (caller's default). DDBM at 50 NFE is
   470:     # the upper-bound reference agents should try to approach with 10x
   471:     # less compute.
   472:     churn_step_ratio = 0.33
   473:     # EDM/Karras-style rho=7 schedule for this reference sampler.
```

### `ecsi` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 459–470:
   456:     seed=None,
   457:     **kwargs,
   458: ):
   459:     """
   460:     ECSI (Endpoint-Conditioned Stochastic Interpolants) sampler.
   461:     Paper: Zhang et al. arXiv:2410.21553
   462:     ('Exploring the Design Space of Diffusion Bridge Models').
   463:     Code: https://github.com/szhan311/ECSI  (sibm/sampling.py: sample_stoch).
   464: 
   465:     Task-local ECSI-inspired sampler settings:
   466:       * pred_mode = "vp"  (already the dbim-codebase e2h default)
   467:       * sigma_min is set below from the local e2h sweep
   468:       * churn_step_ratio = 0.3
   469:       * rho = 0.6
   470:       * NFE = steps (5 for e2h)
   471: 
   472:     Convention mapping ECSI(alpha,beta,gamma) -> dbim-codebase(b_t,a_t,c_t):
   473:     dbim's x_t = a_t*x_T + b_t*x_0 + c_t*noise, so ECSI's alpha (x_0 coef)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
