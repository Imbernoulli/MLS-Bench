# MLS-Bench: llm-rl-importance-sampling

# LLM Online RL: Importance-Sampling Granularity for Policy Optimization

## Research Question
Design a custom **importance-sampling (IS) strategy** for the clipped policy-gradient loss used in LLM online RL. The advantage estimator (GRPO), reward model, rollout setup, and KL configuration are fixed; the only variable is how the old-policy / current-policy log-probs are converted into ratios, clipped, and aggregated. The goal is improved math-reasoning accuracy and reduced gradient variance.

## Background
In PPO-style LLM RL, the per-token policy objective uses an importance ratio
```
r_{i,t} = exp(log_prob_new(y_{i,t}) − log_prob_old(y_{i,t}))
```
applied to per-token advantages. The granularity of this ratio (and of the clipping) is an open research axis:

- **Token-level IS** (vanilla PPO / GRPO; Schulman et al., PPO, 2017, arXiv:1707.06347; Shao et al., DeepSeekMath, 2024, arXiv:2402.03300). Each token has its own ratio, clipped independently. Variance can be very high for long LLM responses because per-token ratios are noisy and errors compound.
- **Sequence-level IS** — Zheng et al., "Group Sequence Policy Optimization", 2025, arXiv:2507.18071 (GSPO). Single scalar ratio per sequence `s_i = exp( mean_t (log_prob_new − log_prob_old) )`, broadcast to every token; reduces variance and stabilizes MoE RL.
- **Truncated / decoupled-clip IS** — Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025, arXiv:2503.14476. Decoupled (asymmetric) clip-low / clip-high, dynamic sampling, token-length-decoupled clipping; built on verl.
- **CISPO-style clipped IS with stop-grad** — MiniMax Team, "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention", 2025, arXiv:2506.13585. Clip the IS weight inside a stop-gradient so gradients flow through `log π` scaled by a bounded IS weight; no token's gradient is zeroed out.
- **Other variants**: dual-clip PPO, geometric-mean aggregation over groups, per-prompt normalised ratios, etc.

## What you can modify
The `compute_custom_policy_loss()` function in `verl/.../custom_policy_loss.py`. The read-only `core_algos.py` contains `compute_policy_loss_vanilla` (token-level PPO), `compute_policy_loss_gspo` (sequence-level), `compute_policy_loss_dppo_kl`, `compute_policy_loss_clip_cov`, etc., as references.

### Interface contract
```python
@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, response_length)
    log_prob: torch.Tensor,          # (bs, response_length)
    advantages: torch.Tensor,        # (bs, response_length)
    response_mask: torch.Tensor,     # (bs, response_length)
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```
- `loss_agg_mode` — common values `"token-mean"`, `"seq-mean-token-mean"`.
- `config` — `ActorConfig` with `clip_ratio`, optional asymmetric `clip_ratio_low` / `clip_ratio_high` (fall back to `clip_ratio` if `None`), `clip_ratio_c` (dual-clip), and `global_batch_info` kwargs forwarded to `agg_loss`.
- Return `(pg_loss, metrics)` where `metrics` has at least `"actor/pg_clipfrac"` and `"actor/ppo_kl"` as Python floats.

Canonical aggregation pattern (from vanilla):
```python
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

Utilities: `verl_F.masked_mean`, `verl_F.masked_whiten`, `agg_loss`, `torch`. `assert config is not None` and read `config.clip_ratio` (do not hardcode ε). Clamp `log_prob − old_log_prob` to a safe range (e.g., `[-20, 20]`) before `exp` for numerical stability. If your strategy aggregates across the sequence (GSPO-like), use `loss_agg_mode="seq-mean-token-mean"` inside your `agg_loss` call. Apply `rollout_is_weights` multiplicatively on `pg_losses` if not `None` (see vanilla).

## Reference baselines
| Baseline | Granularity | Reference |
| --- | --- | --- |
| `token_level` | per-token ratio + per-token clip | PPO (Schulman et al., 2017) |
| `sequence_level` | sequence-mean log-ratio + sequence clip | GSPO, arXiv:2507.18071 |
| `first_k_tokens` | per-token ratio for first K=64 tokens, stop-grad after | DAPO-style truncated IS, arXiv:2503.14476 |

## Fixed Pipeline
- **Policy**: Qwen2.5-0.5B (full-parameter training), verl framework, GRPO advantage estimator.
- **Training set**: simpleRL-Zoo MATH level 3–5 (Qwen split).
- **Hyperparameters**: 100 PPO steps, 16 rollout samples per prompt, batch size 128, 1 H200 GPU.
- Advantage estimator, reward manager, model, rollout setup, optimizer, and evaluation are all fixed.

## Evaluation
Math-reasoning accuracy (`mean@1`) on **GSM8K**, **MATH-500**, and **AMC 23**; primary score is the mean across the three.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/verl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `verl/verl/trainer/ppo/custom_policy_loss.py`
- editable lines **17–72**




## Readable Context


### `verl/verl/trainer/ppo/custom_policy_loss.py`  [EDITABLE — lines 17–72 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Licensed under the Apache License, Version 2.0
     3: """Custom policy loss / importance-sampling strategy for verl PPO training."""
     4: 
     5: from typing import Any, Optional
     6: 
     7: import torch
     8: 
     9: import verl.utils.torch_functional as verl_F
    10: from verl.workers.config import ActorConfig
    11: from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
    12: 
    13: # =====================================================================
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: 
    18: @register_policy_loss("custom")
    19: def compute_custom_policy_loss(
    20:     old_log_prob: torch.Tensor,
    21:     log_prob: torch.Tensor,
    22:     advantages: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     loss_agg_mode: str = "token-mean",
    25:     config: Optional[ActorConfig] = None,
    26:     rollout_is_weights: torch.Tensor | None = None,
    27: ) -> tuple[torch.Tensor, dict[str, Any]]:
    28:     """Compute the clipped policy objective for LLM online RL.
    29: 
    30:     This function is called by the verl training loop.  The core design
    31:     axis is *importance-sampling granularity*: how the ratio
    32:         r = exp(log_prob - old_log_prob)
    33:     is formed and clipped (per-token, per-sequence, truncated to a
    34:     prefix, etc.).  See GSPO (Zheng et al., 2025, arXiv:2507.18071),
    35:     DAPO (arXiv:2503.14476), and CISPO (MiniMax M1, arXiv:2506.13585)
    36:     for references.
    37: 
    38:     Args:
    39:         old_log_prob: (bs, response_length)
    40:             Log-probabilities of each token under the old (rollout) policy.
    41:         log_prob: (bs, response_length)
    42:             Log-probabilities of each token under the current policy.
    43:         advantages: (bs, response_length)
    44:             Per-token advantage estimates.
    45:         response_mask: (bs, response_length)
    46:             Binary mask (1 = valid response token).
    47:         loss_agg_mode: Aggregation mode forwarded to ``agg_loss``.
    48:             Typical values: "token-mean", "seq-mean-token-mean".
    49:         config: ``ActorConfig`` with fields such as ``clip_ratio``,
    50:             ``clip_ratio_low``, ``clip_ratio_high``, and
    51:             ``global_batch_info`` (passed as kwargs to ``agg_loss``).
    52:             ``config.get("name", default)`` is supported for optional
    53:             fields like ``clip_ratio_c``.
    54:         rollout_is_weights: Optional per-token rollout-correction weights.
    55: 
    56:     Returns:
    57:         pg_loss: scalar policy-gradient loss tensor.
    58:         metrics: dict with at least ``actor/pg_clipfrac`` and
    59:             ``actor/ppo_kl`` as Python floats.
    60: 
    61:     Typical call to aggregate:
    62:         pg_loss = agg_loss(
    63:             loss_mat=pg_losses,
    64:             loss_mask=response_mask,
    65:             loss_agg_mode=loss_agg_mode,
    66:             **config.global_batch_info,
    67:         )
    68:     """
    69:     raise NotImplementedError(
    70:         "Implement your custom importance-sampling policy loss here. "
    71:         "See core_algos.py for reference (compute_policy_loss_vanilla / gspo)."
    72:     )
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **deepmath-3bench-h100** — wall-clock budget `6:00:00`, compute share `2`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `token_level` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–66:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """Token-level vanilla PPO: per-token ratio, per-token clip."""
    31:     assert config is not None
    32:     clip_ratio = config.clip_ratio
    33:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    34:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    35:     clip_ratio_c = config.get("clip_ratio_c", 3.0)
    36:     assert clip_ratio_c > 1.0
    37: 
    38:     negative_approx_kl = log_prob - old_log_prob
    39:     negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    40:     ratio = torch.exp(negative_approx_kl)
    41:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    42: 
    43:     pg_losses1 = -advantages * ratio
    44:     pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    45:     clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    46:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    47: 
    48:     pg_losses3 = -advantages * clip_ratio_c
    49:     clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    50:     pg_clipfrac_lower = verl_F.masked_mean(
    51:         torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    52:     )
    53:     pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    54: 
    55:     if rollout_is_weights is not None:
    56:         pg_losses = pg_losses * rollout_is_weights
    57: 
    58:     pg_loss = agg_loss(
    59:         loss_mat=pg_losses, loss_mask=response_mask,
    60:         loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    61:     )
    62:     return pg_loss, {
    63:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    64:         "actor/ppo_kl": ppo_kl.detach().item(),
    65:         "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    66:     }
```

### `sequence_level` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–62:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """Sequence-level IS (GSPO): one scalar ratio per sequence."""
    31:     assert config is not None
    32:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    33:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    34: 
    35:     negative_approx_kl = log_prob - old_log_prob
    36:     seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    37:     neg_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    38: 
    39:     # straight-through: keep per-token log_prob gradient, ratio value is per-sequence
    40:     log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    41:     log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    42:     seq_ratio = torch.exp(log_seq_ratio)
    43: 
    44:     pg_losses1 = -advantages * seq_ratio
    45:     pg_losses2 = -advantages * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    46:     pg_losses = torch.maximum(pg_losses1, pg_losses2)
    47: 
    48:     if rollout_is_weights is not None:
    49:         pg_losses = pg_losses * rollout_is_weights
    50: 
    51:     # GSPO aggregates at the sequence level (seq-mean-token-mean)
    52:     pg_loss = agg_loss(
    53:         loss_mat=pg_losses, loss_mask=response_mask,
    54:         loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    55:     )
    56:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    57:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    58:     return pg_loss, {
    59:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    60:         "actor/ppo_kl": ppo_kl.detach().item(),
    61:         "actor/pg_clipfrac_lower": 0.0,
    62:     }
```

### `first_k_tokens` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–65:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """First-K truncated IS: per-token ratio for t<K, detached for t>=K."""
    31:     assert config is not None
    32:     K = 64  # prefix length with live IS gradient
    33:     clip_ratio = config.clip_ratio
    34:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    35:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    36: 
    37:     negative_approx_kl = log_prob - old_log_prob
    38:     negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    39:     ratio = torch.exp(negative_approx_kl)
    40:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    41: 
    42:     # Build prefix mask: 1 for the first K response positions, 0 afterwards.
    43:     T = ratio.shape[-1]
    44:     positions = torch.arange(T, device=ratio.device).unsqueeze(0)  # (1, T)
    45:     prefix_mask = (positions < K).to(ratio.dtype)                  # (1, T)
    46:     # Detach ratio beyond prefix: ratio_eff = ratio*prefix + detach(ratio)*(1-prefix)
    47:     ratio_eff = ratio * prefix_mask + ratio.detach() * (1.0 - prefix_mask)
    48: 
    49:     pg_losses1 = -advantages * ratio_eff
    50:     pg_losses2 = -advantages * torch.clamp(ratio_eff, 1 - clip_ratio_low, 1 + clip_ratio_high)
    51:     pg_losses = torch.maximum(pg_losses1, pg_losses2)
    52: 
    53:     if rollout_is_weights is not None:
    54:         pg_losses = pg_losses * rollout_is_weights
    55: 
    56:     pg_loss = agg_loss(
    57:         loss_mat=pg_losses, loss_mask=response_mask,
    58:         loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    59:     )
    60:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    61:     return pg_loss, {
    62:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    63:         "actor/ppo_kl": ppo_kl.detach().item(),
    64:         "actor/pg_clipfrac_lower": 0.0,
    65:     }
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
