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
