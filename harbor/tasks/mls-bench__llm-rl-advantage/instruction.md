# MLS-Bench: llm-rl-advantage

# LLM Online RL: Advantage Estimation for GRPO-Style Training

## Research Question
Design a custom advantage estimator for online RL fine-tuning of an LLM. Given per-token rewards, response masks, and group identifiers (multiple sampled responses per prompt), output per-token advantages and returns that the PPO/GRPO actor loss will use. The goal is to improve sample efficiency and policy-learning stability for math reasoning.

## Background
In LLM RL (RLHF / RL with verifiable rewards), the advantage estimator decides how each response is weighted in the policy gradient. Common design choices include:

- **GRPO** — Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024, arXiv:2402.03300. Sample G responses per prompt, compute `(reward - group_mean) / (group_std + eps)`, broadcast to all tokens. Critic-free.
- **Dr. GRPO** — Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective", 2025, arXiv:2503.20783. Removes the std-normalization (group-mean baseline only) to fix a length / question-difficulty bias in GRPO.
- **REINFORCE++ / REINFORCE++-baseline** — Hu, "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization", 2025, arXiv:2501.03262. Uses global / batch-level advantage whitening; the `-baseline` variant combines a per-prompt group baseline with batch-level token whitening for reasoning tasks.
- **RLOO (REINFORCE Leave-One-Out)** — Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", ICLR 2024, arXiv:2402.14740. For each response, baseline = mean reward of the *other* responses in the group: `r_i − mean(r_{j≠i})`.
- **Outcome-level vs token-level**: most estimators broadcast a per-sequence advantage to all tokens; token-level methods assign different advantages per position (e.g., REINFORCE++ token-level discounted returns).

## What you can modify
The `compute_custom_advantage()` function in `verl/verl/trainer/ppo/custom_advantage.py`. The read-only reference file `core_algos.py` contains 13 built-in estimators (GRPO, RLOO, REINFORCE++, REINFORCE++-baseline, OPO, REMAX, GPG, …) you may study.

### Interface contract
The training loop calls your function via the verl advantage-estimator registry:

```python
@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray = None,           # (bs,) group ID per sample
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    ref_log_probs: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:  # (advantages, returns)
```

- `token_level_rewards` — per-token rewards; for outcome rewards the scalar is at the last valid token (`.sum(dim=-1)` for per-sequence score).
- `response_mask` — binary validity mask.
- `index` — group / prompt identifier; same index = same prompt (16 responses per prompt).
- `config` — `AlgoConfig` with `gamma`, `lam`, `norm_adv_by_std_in_grpo`, etc.
- `old_log_probs`, `ref_log_probs` — per-token log-probs under the rollout / reference policy. Per-token KL ≈ `old_log_probs − ref_log_probs`.
- Both returned tensors are `(bs, response_length)` and must be masked by `response_mask`.

Available utilities: `verl_F.masked_whiten(values, mask)`, `verl_F.masked_mean(values, mask)`; `defaultdict`, `torch`, `numpy`. Computation should be wrapped in `torch.no_grad()`. For outcome-level estimators, broadcast the per-sequence advantage to all tokens.

## Reference baselines
- `grpo` — group mean + group std (std-normalized).
- `dr_grpo` — group mean only (no std).
- `reinforce_plus_plus_baseline` — group mean + token-level batch whitening.

## Fixed Pipeline
- **Policy**: Qwen2.5-0.5B (full-parameter training).
- **Framework**: verl.
- **Training set**: simpleRL-Zoo MATH level 3–5 (Qwen split), ~8K problems.
- **RL hyperparameters**: 100 steps, 16 rollout samples per prompt, batch size 128, 1 H200 GPU per experiment.
- The reward manager, model, rollout config, optimizer, KL-loss setting, and evaluation data are all fixed.

## Evaluation
Math-reasoning accuracy (`mean@1`) on:
- **GSM8K** — grade-school math (1,319 problems).
- **MATH-500** — 500-problem subset of MATH competition problems.
- **AMC 23** — AMC 2022–2023 competition-math subset.

Higher is better.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/verl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `verl/verl/trainer/ppo/custom_advantage.py`
- editable lines **17–72**




## Readable Context


### `verl/verl/trainer/ppo/custom_advantage.py`  [EDITABLE — lines 17–72 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Licensed under the Apache License, Version 2.0
     3: """Custom advantage estimator for verl PPO training."""
     4: 
     5: from collections import defaultdict
     6: from typing import Optional
     7: 
     8: import numpy as np
     9: import torch
    10: 
    11: import verl.utils.torch_functional as verl_F
    12: from verl.trainer.config import AlgoConfig
    13: from verl.trainer.ppo.core_algos import register_adv_est
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom advantage estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: @register_adv_est("custom")
    21: def compute_custom_advantage(
    22:     token_level_rewards: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     index: np.ndarray = None,
    25:     epsilon: float = 1e-6,
    26:     config: Optional[AlgoConfig] = None,
    27:     old_log_probs: Optional[torch.Tensor] = None,
    28:     ref_log_probs: Optional[torch.Tensor] = None,
    29:     **kwargs,
    30: ) -> tuple[torch.Tensor, torch.Tensor]:
    31:     """Compute advantage estimates for policy optimization.
    32: 
    33:     This function is called by the verl training loop to compute per-token
    34:     advantage and return estimates.  You may implement any strategy — e.g.
    35:     group-normalized (GRPO), leave-one-out (RLOO), discounted cumulative
    36:     returns (REINFORCE++), KL-penalized methods, or a novel combination.
    37: 
    38:     Args:
    39:         token_level_rewards: (bs, response_length)
    40:             Per-token rewards.  For outcome-based rewards the scalar reward
    41:             is placed at the last valid token; use .sum(dim=-1) to recover
    42:             per-sequence scores.
    43:         response_mask: (bs, response_length)
    44:             Binary mask indicating valid response tokens (1 = valid).
    45:         index: (bs,) optional
    46:             Group/prompt identifier for each sample.  Samples sharing the
    47:             same index were generated from the same prompt and can be
    48:             compared for group-based normalization.
    49:         epsilon: Small constant to avoid division by zero.
    50:         config: Algorithm configuration (AlgoConfig dataclass).
    51:             Useful fields: config.gamma (discount factor, default 1.0),
    52:             config.lam (GAE lambda), config.norm_adv_by_std_in_grpo, etc.
    53:         old_log_probs: (bs, response_length) optional
    54:             Log-probabilities of each token under the current rollout policy.
    55:             Useful for entropy-based exploration bonuses, importance weighting,
    56:             or probability-aware advantage shaping.
    57:         ref_log_probs: (bs, response_length) optional
    58:             Log-probabilities under the reference (frozen) policy.
    59:             Useful for KL-penalized advantage: KL ≈ old_log_probs - ref_log_probs.
    60: 
    61:     Returns:
    62:         advantages: (bs, response_length) — advantage estimates per token.
    63:         returns:    (bs, response_length) — return estimates per token.
    64: 
    65:     Available utilities:
    66:         verl_F.masked_whiten(values, mask)  — zero-mean unit-variance whitening
    67:         verl_F.masked_mean(values, mask)    — masked mean
    68:     """
    69:     raise NotImplementedError(
    70:         "Implement your custom advantage estimator here. "
    71:         "See core_algos.py for reference implementations (GRPO, RLOO, REINFORCE++, etc.)."
    72:     )
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `grpo` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_advantage.py`:

```python
Lines 17–65:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom advantage estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: @register_adv_est("custom")
    21: def compute_custom_advantage(
    22:     token_level_rewards: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     index: np.ndarray = None,
    25:     epsilon: float = 1e-6,
    26:     config: Optional[AlgoConfig] = None,
    27:     **kwargs,
    28: ) -> tuple[torch.Tensor, torch.Tensor]:
    29:     """GRPO: Group Relative Policy Optimization advantage estimator.
    30: 
    31:     Computes outcome-level advantages by normalizing rewards within each
    32:     prompt group by the group mean and standard deviation.
    33:     """
    34:     scores = token_level_rewards.sum(dim=-1)
    35: 
    36:     id2score = defaultdict(list)
    37:     id2mean = {}
    38:     id2std = {}
    39: 
    40:     norm_adv_by_std = True
    41:     if config is not None:
    42:         norm_adv_by_std = getattr(config, "norm_adv_by_std_in_grpo", True)
    43: 
    44:     with torch.no_grad():
    45:         bsz = scores.shape[0]
    46:         for i in range(bsz):
    47:             id2score[index[i]].append(scores[i])
    48:         for idx in id2score:
    49:             if len(id2score[idx]) == 1:
    50:                 id2mean[idx] = torch.tensor(0.0)
    51:                 id2std[idx] = torch.tensor(1.0)
    52:             elif len(id2score[idx]) > 1:
    53:                 scores_tensor = torch.stack(id2score[idx])
    54:                 id2mean[idx] = torch.mean(scores_tensor)
    55:                 id2std[idx] = torch.std(scores_tensor)
    56:             else:
    57:                 raise ValueError(f"no score in prompt index: {idx}")
    58:         for i in range(bsz):
    59:             if norm_adv_by_std:
    60:                 scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    61:             else:
    62:                 scores[i] = scores[i] - id2mean[index[i]]
    63:         scores = scores.unsqueeze(-1) * response_mask
    64: 
    65:     return scores, scores
```

### `dr_grpo` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_advantage.py`:

```python
Lines 17–55:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom advantage estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: @register_adv_est("custom")
    21: def compute_custom_advantage(
    22:     token_level_rewards: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     index: np.ndarray = None,
    25:     epsilon: float = 1e-6,
    26:     config: Optional[AlgoConfig] = None,
    27:     **kwargs,
    28: ) -> tuple[torch.Tensor, torch.Tensor]:
    29:     """Dr. GRPO: GRPO without standard deviation normalization.
    30: 
    31:     Computes outcome-level advantages by subtracting the group mean reward,
    32:     without dividing by the group standard deviation.
    33:     """
    34:     scores = token_level_rewards.sum(dim=-1)
    35: 
    36:     id2score = defaultdict(list)
    37:     id2mean = {}
    38: 
    39:     with torch.no_grad():
    40:         bsz = scores.shape[0]
    41:         for i in range(bsz):
    42:             id2score[index[i]].append(scores[i])
    43:         for idx in id2score:
    44:             if len(id2score[idx]) == 1:
    45:                 id2mean[idx] = torch.tensor(0.0)
    46:             elif len(id2score[idx]) > 1:
    47:                 scores_tensor = torch.stack(id2score[idx])
    48:                 id2mean[idx] = torch.mean(scores_tensor)
    49:             else:
    50:                 raise ValueError(f"no score in prompt index: {idx}")
    51:         for i in range(bsz):
    52:             scores[i] = scores[i] - id2mean[index[i]]
    53:         scores = scores.unsqueeze(-1) * response_mask
    54: 
    55:     return scores, scores
```

### `reinforce_plus_plus_baseline` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_advantage.py`:

```python
Lines 17–59:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom advantage estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: @register_adv_est("custom")
    21: def compute_custom_advantage(
    22:     token_level_rewards: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     index: np.ndarray = None,
    25:     epsilon: float = 1e-6,
    26:     config: Optional[AlgoConfig] = None,
    27:     **kwargs,
    28: ) -> tuple[torch.Tensor, torch.Tensor]:
    29:     """REINFORCE++-baseline: group-centered reward, token-level batch whitening.
    30: 
    31:     Subtract per-prompt group mean from each response's scalar reward,
    32:     broadcast to token level, then masked-whiten over all valid response
    33:     tokens in the batch (longer responses contribute more to the whitening
    34:     statistics).
    35:     """
    36:     response_length = token_level_rewards.shape[-1]
    37:     scores = token_level_rewards.sum(dim=-1)
    38: 
    39:     id2score = defaultdict(list)
    40:     id2mean = {}
    41: 
    42:     with torch.no_grad():
    43:         bsz = scores.shape[0]
    44:         for i in range(bsz):
    45:             id2score[index[i]].append(scores[i])
    46:         for idx in id2score:
    47:             if len(id2score[idx]) == 1:
    48:                 id2mean[idx] = torch.tensor(0.0)
    49:             elif len(id2score[idx]) > 1:
    50:                 id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
    51:             else:
    52:                 raise ValueError(f"no score in prompt index: {idx}")
    53:         for i in range(bsz):
    54:             scores[i] = scores[i] - id2mean[index[i]]
    55: 
    56:         scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    57:         scores = verl_F.masked_whiten(scores, response_mask) * response_mask
    58: 
    59:     return scores, scores
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
