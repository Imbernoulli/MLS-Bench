# MLS-Bench: llm-rl-reward-normalization

# LLM Online RL: Reward Normalization Before Advantage Estimation

## Research Question
Design a **reward-normalization** strategy that runs **before** advantage estimation in LLM RL. The reward manager produces a per-response scalar; this transformation reshapes the per-token reward tensor that GRPO will then consume. The goal is to improve gradient scale and variance for math reasoning without erasing useful signal or double-normalizing destructively against GRPO.

## Background
In GRPO-style LLM RL, the reward manager produces a per-response scalar (e.g., 1.0 for a correct answer, 0.0 otherwise, possibly with a format bonus). This scalar is placed at the last valid token, becoming a `(batch_size, response_length)` tensor `token_level_scores`. The downstream GRPO advantage estimator then subtracts a per-prompt baseline and (optionally) divides by the per-prompt group std. The quality of that baseline depends on the **scale and distribution of the rewards going in**.

Common design choices for upstream reward normalization:

- **Raw / outcome-only** — no normalization (verl default).
- **Batch-std whitening** — subtract batch mean, divide by batch std + eps. Classic RLHF baseline (Ouyang et al., "Training language models to follow instructions with human feedback", 2022, arXiv:2203.02155).
- **Group-std (GRPO-style)** — subtract per-prompt group mean, divide by per-prompt group std at the reward stage (same statistic GRPO uses downstream).
- **Length-aware** — divide the scalar by a function of response length (e.g., `√T`) before broadcasting; motivated by DAPO's observation (Yu et al., 2025, arXiv:2503.14476) that longer responses accumulate more per-token gradient signal, biasing the policy toward verbose outputs.
- **Percentile clipping** — clip to robust quantiles (e.g., 5th–95th) before normalization to limit outlier influence.

## What you can modify
The `normalize_rewards()` function in `custom_reward_normalization.py`. The read-only `core_algos.py` contains verl's advantage estimators (GRPO, REINFORCE++, Dr.GRPO, RLOO, …); your normalization runs upstream of all of them.

### Interface contract
```python
def normalize_rewards(
    token_level_scores: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray = None,           # (bs,) group/prompt identifier
    epsilon: float = 1e-6,
    config: Optional[object] = None,    # algorithm hydra config
    **kwargs,
) -> torch.Tensor:                      # (bs, response_length) normalized
```
- Outcome rewards live at the last valid token; use `.sum(dim=-1)` to recover per-sequence scalars.
- Samples sharing `index[i]` come from the same prompt (16 rollouts per prompt).
- Output shape must equal input shape; multiply by `response_mask` to preserve "outcome reward at last token" semantics where appropriate.
- Wrap in `torch.no_grad()`.
- Available utilities: `verl_F.masked_whiten`, `verl_F.masked_mean`, `defaultdict`, `torch`, `numpy`.
- This runs **before** the advantage estimator. GRPO will still subtract the group mean and divide by group std on top of your output — design with that interaction in mind.

## Reference baselines
| Baseline | Strategy |
|---|---|
| `outcome_only` | no normalization (verl default) |
| `group_std` | per-prompt group mean + std normalization at reward stage |
| `batch_std` | batch-mean + batch-std whitening over valid tokens (RLHF-style) |
| `length_aware` | divide scalar by `√(response_length)` before broadcast (DAPO length-bias fix) |

## Fixed Pipeline
- **Policy**: Qwen2.5-0.5B (full-parameter), verl, GRPO advantage estimator, n=16 rollouts per prompt.
- **Training set**: simpleRL-Zoo MATH level 3–5 (Qwen split) + 5K DeepMath problems.
- **Hyperparameters**: 100 steps, batch size 128, max response length 16,384 tokens, `test_freq=25`, `total_epochs=1`.
- Reward source, advantage estimator, model, rollout setup, optimizer, KL-loss setting, and evaluation data are fixed.

## Evaluation
Math-reasoning accuracy (`mean@1`) on **GSM8K**, **MATH-500**, and **AMC 23**; primary score is the mean across the three.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/verl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `verl/verl/trainer/ppo/custom_reward_normalization.py`
- editable lines **17–72**




## Readable Context


### `verl/verl/trainer/ppo/custom_reward_normalization.py`  [EDITABLE — lines 17–72 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Licensed under the Apache License, Version 2.0
     3: """Custom reward-normalization for verl PPO training.
     4: 
     5: This function runs UPSTREAM of advantage estimation: it transforms the
     6: raw scalar reward(s) assigned by the reward manager into normalized
     7: per-token rewards, which then feed into advantage computation.
     8: """
     9: 
    10: from collections import defaultdict
    11: from typing import Optional
    12: 
    13: import numpy as np
    14: import torch
    15: 
    16: import verl.utils.torch_functional as verl_F
    17: 
    18: # =====================================================================
    19: # EDITABLE: Implement your custom reward normalization below.
    20: # =====================================================================
    21: 
    22: 
    23: def normalize_rewards(
    24:     token_level_scores: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     index: Optional[np.ndarray] = None,
    27:     epsilon: float = 1e-6,
    28:     config: Optional[object] = None,
    29:     **kwargs,
    30: ) -> torch.Tensor:
    31:     """Normalize per-response scalar rewards before advantage estimation.
    32: 
    33:     This is the UPSTREAM hook: it receives the raw reward tensor produced
    34:     by the reward manager (outcome reward placed at the last valid token)
    35:     and returns a transformed reward tensor of the same shape.  The result
    36:     is written back into ``data.batch["token_level_scores"]`` and then
    37:     consumed by the advantage estimator (GRPO / REINFORCE++ / GAE / ...).
    38: 
    39:     Args:
    40:         token_level_scores: (bs, response_length)
    41:             Raw per-token reward.  For outcome rewards only the last valid
    42:             token is non-zero; ``token_level_scores.sum(dim=-1)`` recovers
    43:             the per-sequence scalar.
    44:         response_mask: (bs, response_length) binary mask (1 = valid token).
    45:         index: (bs,) optional — group/prompt identifier.  Samples sharing
    46:             the same index were generated from the same prompt (GRPO-style
    47:             group of n rollouts).  Use ``defaultdict(list)`` to collect
    48:             per-group statistics.
    49:         epsilon: small constant to avoid division by zero.
    50:         config: the full ``algorithm`` hydra config (DictConfig). Useful
    51:             if you add per-strategy hyperparameters.
    52: 
    53:     Returns:
    54:         token_level_scores: (bs, response_length) — normalized rewards,
    55:         masked by ``response_mask``.  Must have the same shape as input.
    56: 
    57:     Available utilities:
    58:         verl_F.masked_whiten(values, mask)  — zero-mean unit-variance
    59:         verl_F.masked_mean(values, mask)    — masked mean
    60:     """
    61:     # Default: raw / identity — pass rewards through unchanged.  This
    62:     # matches verl's current (unnormalized) reward handling; the GRPO
    63:     # default advantage estimator already does its own group-std
    64:     # normalization on top.  Replace this body to experiment with other
    65:     # reward-space normalization strategies.
    66:     #
    67:     # To implement a new strategy, replace this body — e.g. batch-std
    68:     # whitening, per-prompt group-std (GRPO), length-aware division,
    69:     # percentile clipping, etc.  See task_description.md for references.
    70:     with torch.no_grad():
    71:         scores = token_level_scores * response_mask
    72:         return scores
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `outcome_only` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_reward_normalization.py`:

```python
Lines 17–36:
    14: import torch
    15: 
    16: import verl.utils.torch_functional as verl_F
    17: # =====================================================================
    18: 
    19: 
    20: def normalize_rewards(
    21:     token_level_scores,
    22:     response_mask,
    23:     index=None,
    24:     epsilon: float = 1e-6,
    25:     config=None,
    26:     **kwargs,
    27: ):
    28:     """outcome_only: no reward-space normalization.
    29: 
    30:     Pass the raw (bs, response_length) reward tensor straight through.
    31:     The outcome reward is left at the last valid token and whatever
    32:     downstream advantage estimator is configured (GRPO by default)
    33:     applies its own normalization.
    34:     """
    35:     with torch.no_grad():
    36:         return token_level_scores * response_mask
```

### `group_std` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_reward_normalization.py`:

```python
Lines 17–61:
    14: import torch
    15: 
    16: import verl.utils.torch_functional as verl_F
    17: # =====================================================================
    18: 
    19: 
    20: def normalize_rewards(
    21:     token_level_scores,
    22:     response_mask,
    23:     index=None,
    24:     epsilon: float = 1e-6,
    25:     config=None,
    26:     **kwargs,
    27: ):
    28:     """group_std: per-prompt group mean + std normalization (GRPO-style)."""
    29:     with torch.no_grad():
    30:         bsz, seq_len = token_level_scores.shape
    31:         scores = token_level_scores.sum(dim=-1)  # (bs,)
    32: 
    33:         if index is None:
    34:             # Fallback to batch-level normalization if no grouping info.
    35:             mean = scores.mean()
    36:             std = scores.std(unbiased=False)
    37:             scores = (scores - mean) / (std + epsilon)
    38:         else:
    39:             id2score = defaultdict(list)
    40:             id2mean = {}
    41:             id2std = {}
    42:             for i in range(bsz):
    43:                 id2score[index[i]].append(scores[i])
    44:             for idx, vs in id2score.items():
    45:                 if len(vs) == 1:
    46:                     id2mean[idx] = torch.tensor(0.0, device=scores.device)
    47:                     id2std[idx] = torch.tensor(1.0, device=scores.device)
    48:                 else:
    49:                     stacked = torch.stack(vs)
    50:                     id2mean[idx] = stacked.mean()
    51:                     id2std[idx] = stacked.std(unbiased=False)
    52:             for i in range(bsz):
    53:                 scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    54: 
    55:         # Place the normalized scalar back at the last valid token of each
    56:         # response so the outcome-reward semantics are preserved.
    57:         out = torch.zeros_like(token_level_scores)
    58:         last_idx = response_mask.long().sum(dim=-1) - 1  # (bs,)
    59:         last_idx = last_idx.clamp(min=0)
    60:         out[torch.arange(bsz, device=out.device), last_idx] = scores
    61:         return out * response_mask
```

### `batch_std` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_reward_normalization.py`:

```python
Lines 17–47:
    14: import torch
    15: 
    16: import verl.utils.torch_functional as verl_F
    17: # =====================================================================
    18: 
    19: 
    20: def normalize_rewards(
    21:     token_level_scores,
    22:     response_mask,
    23:     index=None,
    24:     epsilon: float = 1e-6,
    25:     config=None,
    26:     **kwargs,
    27: ):
    28:     """batch_std: subtract batch mean, divide by batch std + eps."""
    29:     with torch.no_grad():
    30:         bsz, seq_len = token_level_scores.shape
    31:         scores = token_level_scores.sum(dim=-1)  # (bs,)
    32: 
    33:         if bsz <= 1:
    34:             # Degenerate case — no normalization possible.
    35:             mean = torch.tensor(0.0, device=scores.device)
    36:             std = torch.tensor(1.0, device=scores.device)
    37:         else:
    38:             mean = scores.mean()
    39:             std = scores.std(unbiased=False)
    40: 
    41:         scores = (scores - mean) / (std + epsilon)
    42: 
    43:         out = torch.zeros_like(token_level_scores)
    44:         last_idx = response_mask.long().sum(dim=-1) - 1  # (bs,)
    45:         last_idx = last_idx.clamp(min=0)
    46:         out[torch.arange(bsz, device=out.device), last_idx] = scores
    47:         return out * response_mask
```

### `length_aware` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_reward_normalization.py`:

```python
Lines 17–41:
    14: import torch
    15: 
    16: import verl.utils.torch_functional as verl_F
    17: # =====================================================================
    18: 
    19: 
    20: def normalize_rewards(
    21:     token_level_scores,
    22:     response_mask,
    23:     index=None,
    24:     epsilon: float = 1e-6,
    25:     config=None,
    26:     **kwargs,
    27: ):
    28:     """length_aware: divide scalar reward by sqrt(response_length)."""
    29:     with torch.no_grad():
    30:         bsz, seq_len = token_level_scores.shape
    31:         scores = token_level_scores.sum(dim=-1)  # (bs,)
    32: 
    33:         lengths = response_mask.sum(dim=-1).to(scores.dtype)  # (bs,)
    34:         denom = torch.sqrt(lengths.clamp(min=1.0)) + epsilon
    35:         scores = scores / denom
    36: 
    37:         out = torch.zeros_like(token_level_scores)
    38:         last_idx = response_mask.long().sum(dim=-1) - 1  # (bs,)
    39:         last_idx = last_idx.clamp(min=0)
    40:         out[torch.arange(bsz, device=out.device), last_idx] = scores
    41:         return out * response_mask
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
