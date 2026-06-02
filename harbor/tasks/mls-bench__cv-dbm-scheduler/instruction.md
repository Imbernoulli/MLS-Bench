# MLS-Bench: cv-dbm-scheduler

# Time Scheduler for Diffusion Bridge Models (NFE = 5)

## Objective

Design a novel time schedule for diffusion bridge sampling that improves
generation quality (lower FID) over standard schedules under an extremely
small denoiser-call budget (NFE = 5).

## Background

In diffusion bridge sampling, the time schedule controls where the sampler
spends its few denoiser evaluations along the bridge trajectory. With only a
handful of steps, schedule shape strongly affects endpoint fidelity, output
diversity, and artifact removal — even when the sampler update rule itself is
held fixed. Common reference schedules include:

- **Uniform / linear** (every step is the same length in `t`).
- **Karras** rho-schedule from EDM (Karras et al., NeurIPS 2022,
  arXiv:2206.00364): `σ_t = ((1 − u) σ_min^{1/ρ} + u σ_max^{1/ρ})^ρ` with
  `ρ = 7`, allocating more resolution to high-noise regions.
- **Cosine** schedules (e.g., from Nichol & Dhariwal, ICML 2021,
  arXiv:2102.09672), originally proposed for the noise schedule but also used
  to derive sampling timesteps.
- **Log-linear** schedules, equally spaced in `log t`, concentrating steps
  near the data endpoint.

Diffusion bridges (Zhou et al. DDBM, arXiv:2309.16948; Zheng et al. DBIM,
arXiv:2405.15885) inherit this design choice: with NFE = 5, picking a good
schedule can change FID substantially without touching the sampler.

## Implementation Contract

⚠️ Implement the schedule inside the function `get_sigmas_uniform` in the
provided file. Despite the legacy name, the intended contribution is a
non-trivial schedule curve, **not** a basic linear / uniform schedule.

```python
import torch

def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    """
    Requirements:
      1. Length:        return a 1D torch.Tensor of exactly length n + 1.
      2. Monotonicity:  the sequence must strictly decrease from t_max to t_min.
      3. Terminal:      the final element (index n) must equal t_min exactly.
      4. Device:        the returned tensor must live on the requested device.
    """
    # For this task, n = 5 (NFE = 5).
    # Implement your schedule formulation here.
    ...
```

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `uniform`   | Linearly spaced timesteps from `t_max` to `t_min`. |
| `karras`    | EDM rho-schedule (Karras et al., NeurIPS 2022, arXiv:2206.00364), `ρ = 7`. |
| `cosine`    | Cosine-spaced timesteps. |
| `loglinear` | Log-linearly spaced timesteps from `t_max` to `t_min`. |

## Evaluation

The benchmark runs the bridge sampler with the candidate schedule on
image-to-image workloads. The active budget is **NFE = 5** denoiser calls per
sample. Metric: **FID**, lower is better.

The proposed schedule should generalize across workloads rather than encode
constants tuned for a single dataset. Do not change the sampler update rule,
the number of allowed denoiser calls, dataset handling, or metric computation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/dbim-codebase/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `dbim-codebase/ddbm/karras_diffusion.py`
- editable lines **310–320**




## Readable Context


### `dbim-codebase/ddbm/karras_diffusion.py`  [EDITABLE — lines 310–320 only]

```python
Lines 301-311:
   301:         x_0.clamp(-1, 1),
   302:         [x.clamp(-1, 1) for x in path],
   303:         nfe,
   304:         [x.clamp(-1, 1) for x in pred_x0],
   305:         sigmas,
   306:         noise,
   307:     )
   308: 
   309: 
   310: def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
   311:     """
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `karras` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 310–320:
   307:     )
   308: 
   309: 
   310: def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
   311:     rho = 7.0
   312:     ramp = torch.linspace(0, 1, n + 1)
   313:     min_inv_rho = t_min ** (1 / rho)
   314:     max_inv_rho = t_max ** (1 / rho)
   315:     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
   316:     sigmas[-1] = t_min  # ensure exact terminal value
   317:     return sigmas.to(device)
   318: 
   319: @torch.no_grad()
   320: def sample_dbim_high_order(
   321:     denoiser,
   322:     diffusion,
   323:     x,
```

### `uniform` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 310–320:
   307:     )
   308: 
   309: 
   310: def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
   311:     return torch.linspace(t_max, t_min, n + 1).to(device)
   312: 
   313: @torch.no_grad()
   314: def sample_dbim_high_order(
   315:     denoiser,
   316:     diffusion,
   317:     x,
   318:     ts,
   319:     mask=None,
   320:     order=2,
   321:     lower_order_final=True,
   322:     seed=None,
   323:     **kwargs,
```

### `cosine` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 310–320:
   307:     )
   308: 
   309: 
   310: def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
   311:     import math
   312:     ramp = torch.linspace(0, 1, n + 1)
   313:     cosine_ramp = (1 - torch.cos(ramp * math.pi)) / 2
   314:     sigmas = t_max + (t_min - t_max) * cosine_ramp
   315:     return sigmas.to(device)
   316: 
   317: @torch.no_grad()
   318: def sample_dbim_high_order(
   319:     denoiser,
   320:     diffusion,
   321:     x,
   322:     ts,
   323:     mask=None,
```

### `loglinear` baseline — editable region  [READ-ONLY — reference implementation]

In `dbim-codebase/ddbm/karras_diffusion.py`:

```python
Lines 310–320:
   307:     )
   308: 
   309: 
   310: def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
   311:     import math
   312:     log_max = math.log(t_max)
   313:     log_min = math.log(max(t_min, 1e-10))
   314:     sigmas = torch.exp(torch.linspace(log_max, log_min, n + 1))
   315:     sigmas[-1] = t_min  # ensure exact terminal value
   316:     return sigmas.to(device)
   317: 
   318: @torch.no_grad()
   319: def sample_dbim_high_order(
   320:     denoiser,
   321:     diffusion,
   322:     x,
   323:     ts,
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
