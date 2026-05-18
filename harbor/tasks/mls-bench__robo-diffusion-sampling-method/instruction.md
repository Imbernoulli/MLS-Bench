# MLS-Bench: robo-diffusion-sampling-method

# Robo-Diffusion: Sampling Algorithm Design

## Objective
Design a single efficient diffusion sampler for a fixed DQL-style diffusion policy, maximizing D4RL MuJoCo return at low inference NFE (number of function evaluations).

This task is deliberately about inference-time sampler choice, not policy learning, guidance, or trajectory planning. The trained actor / critic, dataset, environment list, seeds, and evaluation loop are fixed.

## Background
A diffusion policy's wall-clock inference cost is dominated by the number of reverse-process steps. Different ODE / SDE solvers reach a given sample quality at different NFE budgets:
- **DDPM** (Ho, Jain, Abbeel, NeurIPS 2020, arXiv:2006.11239): the original Markovian sampler; high quality but slow.
- **DDIM** (Song, Meng, Ermon, ICLR 2021, arXiv:2010.02502): non-Markovian deterministic sampler that hits comparable quality in 10–50× fewer steps.
- **DPM-Solver++** (Lu et al., 2022, arXiv:2211.01095): high-order ODE solver that reaches strong sample quality at ~10–20 steps for guided DPM sampling.

The setup builds on **CleanDiffuser** (Dong et al., NeurIPS 2024, arXiv:2406.09509) and the underlying actor is a DQL-style diffusion policy (Wang et al., ICLR 2023, arXiv:2208.06193) trained on **D4RL** (Fu et al., 2020, arXiv:2004.07219).

## What You Can Modify
- `solver` in `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`
- `sampling_steps` in the same YAML file

## What Is Fixed
- The pipeline code, model architecture, critic, and training objective
- `diffusion_steps`, training budgets, checkpoint selection, and EMA use
- D4RL environment names, seeds, and vectorized evaluation

The score's NFE term is read from the same `sampling_steps` field passed to CleanDiffuser's sampler. Custom pipeline-code samplers are intentionally out of scope here because they would decouple true NFE from the reported score column.

## Evaluation
Evaluated on three D4RL MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Metrics: `normalized_score` (D4RL return) and `sampling_steps` (NFE per inference call).

### Score formula
The per-env score multiplies a quality term by an NFE penalty:

```
score(env) = sigmoid(normalized_score) * penalty_upper(sampling_steps, target=10)
penalty_upper(x, target=10) = exp(-0.015 * (x - 10))   for x > 10
                              1.0                       for x <= 10
```

NFE penalty cheat-sheet:

| sampling_steps | penalty | example                |
|---------------:|--------:|------------------------|
| 10             | 1.000   | DPM-Solver++ baseline  |
| 20             | 0.861   | DDIM baseline          |
| 50             | 0.549   |                        |
| 100            | 0.259   | DDPM baseline          |

Task score is the geometric mean of the three env scores. **Submitting at lower NFE is strictly preferred when quality is comparable.**

## Baselines

### default
DDPM sampling with 100 steps — standard but slow. This is the unmodified
template baseline (registered as `default` in the config).

### ddim
DDIM sampling with 20 steps — faster deterministic sampling.

### dpm_solver
DPM-Solver++ with 10 steps — fast high-quality sampling.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CleanDiffuser/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`
- editable lines **15–15**
- editable lines **17–17**


Other files you may **read** for context (do not modify):
- `CleanDiffuser/pipelines/custom_sampling_method.py`


## Readable Context


### `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`  [EDITABLE — lines 15–15, lines 17–17 only]

```yaml
     1: defaults:
     2:   - _self_
     3:   - task: hopper-medium-v2
     4: 
     5: pipeline_name: custom_sampling_method
     6: mode: train
     7: seed: 42
     8: device: cuda:0
     9: 
    10: # Environment
    11: normalize_reward: True
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ddpm
    16: diffusion_steps: 100
    17: sampling_steps: 100
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
    21: 
    22: # Critic
    23: hidden_dim: 256
    24: critic_learning_rate: 0.0003
    25: 
    26: # Training
    27: gradient_steps: 100000
    28: batch_size: 256
    29: ema_update_interval: 5
    30: log_interval: 1000
    31: save_interval: 50000
    32: 
    33: # Inference
    34: ckpt: latest
    35: num_envs: 50
    36: num_episodes: 3
    37: num_candidates: 50
    38: temperature: 0.5
    39: use_ema: True
    40: 
    41: # hydra
    42: hydra:
    43:   job:
    44:     chdir: false
```


## Adapter Warnings

Some reference context could not be rendered completely:

- `default` has no edit_ops entry


## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **train_hopper** — wall-clock budget `18:00:00`, compute share `1`
- **train_walker2d** — wall-clock budget `8:00:00`, compute share `1`
- **train_halfcheetah** — wall-clock budget `8:00:00`, compute share `1`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ddim` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`:

```python
Lines 15–15:
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ddim
    16: diffusion_steps: 100
    17: sampling_steps: 20
    18: predict_noise: True

Lines 17–17:
    14: # Actor
    15: solver: ddim
    16: diffusion_steps: 100
    17: sampling_steps: 20
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
```

### `dpm_solver` baseline — editable region  [READ-ONLY — reference implementation]

In `CleanDiffuser/configs/custom/mujoco/mujoco.yaml`:

```python
Lines 15–15:
    12: discount: 0.99
    13: 
    14: # Actor
    15: solver: ode_dpmsolver++_2M
    16: diffusion_steps: 100
    17: sampling_steps: 10
    18: predict_noise: True

Lines 17–17:
    14: # Actor
    15: solver: ode_dpmsolver++_2M
    16: diffusion_steps: 100
    17: sampling_steps: 10
    18: predict_noise: True
    19: ema_rate: 0.995
    20: actor_learning_rate: 0.0003
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
