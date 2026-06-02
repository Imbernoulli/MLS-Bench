# MLS-Bench: llm-rl-kl-estimator

# LLM Online RL: KL-Divergence Estimator for Actor KL Loss

## Research Question
Design a per-token KL-divergence estimator for the actor-side KL-loss term used during LLM RL fine-tuning. The estimator receives current-policy and frozen-reference-policy log-probabilities and returns the KL penalty added to the policy-gradient loss. The goal is to keep the policy close enough to the reference for stable training while avoiding overly noisy or overly conservative penalties on math reasoning.

## Background
When training an LLM policy with PPO/GRPO, a KL penalty keeps the policy close to a frozen reference model (typically the SFT initialization). In verl, KL can enter the objective in two places:

1. **KL-in-reward** (`algorithm.use_kl_in_reward=True`) — per-token KL is subtracted from the reward before advantage estimation.
2. **KL-loss** (`actor.use_kl_loss=True`) — per-token KL is aggregated and added to the policy-gradient loss with `kl_loss_coef`. **This task studies the KL-loss path.**

Because the policy and reference distributions are over the full vocabulary, the exact KL is expensive; practitioners use Monte-Carlo estimators from per-token log-probs. Reference: Schulman, "Approximating KL Divergence", 2020 (http://joschu.net/blog/kl-approx.html).

| Name | Formula | Bias | Variance |
|---|---|---|---|
| `k1` (`kl`) | `log p − log q` | unbiased | high |
| `k2` (`mse`) | `0.5 * (log p − log q)^2` | biased (overestimates) | low |
| `k3` (`low_var_kl`) | `exp(log q − log p) − (log q − log p) − 1` | unbiased | low, always `≥ 0` |
| `abs` | `|log p − log q|` | biased | medium, robust |

Each has different gradient properties — `k2` has exact-KL gradients in expectation; `k1` / `k3` have exact-KL values in expectation but biased gradients. verl also exposes straight-through variants (`k3+`) that combine `k3` forward values with `k2` backward gradients.

## What you can modify
The `compute_custom_kl_penalty()` function in `verl/.../custom_kl_penalty.py`. The read-only `core_algos.py` contains verl's built-in estimators (`k1`/`kl`, `k2`/`mse`, `k3`/`low_var_kl`, `abs`, plus `k3+` straight-through variants).

### Interface contract
```python
def compute_custom_kl_penalty(
    logprob: torch.Tensor,      # (bs, response_length) — current actor
    ref_logprob: torch.Tensor,  # (bs, response_length) — frozen reference
) -> torch.Tensor:              # (bs, response_length) per-token KL
```
- The training loop multiplies your output by `response_mask`, aggregates via `agg_loss(...)`, then multiplies by `kl_loss_coef` and adds to the PG loss.
- Do NOT wrap in `torch.no_grad()` — gradients must flow back to the actor.
- Same shape as inputs.
- Numerical safety: clamp extreme log-ratios before `exp` (as `k3` does).
- Do not touch the wiring section at the bottom of the file (it monkey-patches `verl.trainer.ppo.core_algos.kl_penalty_forward` so `actor.kl_loss_type=custom` dispatches to your function).

## Reference baselines
| Baseline | Formula | Notes |
|---|---|---|
| `k1` | `logprob − ref_logprob` | naive unbiased |
| `k2` | `0.5 * (logprob − ref_logprob)^2` | Schulman MSE |
| `k3` | `exp(r − l) − (r − l) − 1` (with clamps) | verl default `low_var_kl` |
| `abs` | `|logprob − ref_logprob|` | robust |

## Fixed Pipeline
The RL framework, advantage estimator, frozen reference policy, reward manager, rollout, model, optimizer, and evaluation are fixed by the harness and not editable. Your estimator is selected via the `custom` KL-loss branch (the wiring at the bottom of the file dispatches `actor.kl_loss_type=custom` to your function); the per-token KL it returns is the only quantity you change.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/verl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `verl/verl/trainer/ppo/custom_kl_penalty.py`
- editable lines **17–56**




## Readable Context


### `verl/verl/trainer/ppo/custom_kl_penalty.py`  [EDITABLE — lines 17–56 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Licensed under the Apache License, Version 2.0
     3: """Custom KL-divergence estimator for verl PPO training (actor KL loss).
     4: 
     5: Registers a "custom" branch on ``core_algos.kl_penalty_forward`` so that
     6: passing ``actor_rollout_ref.actor.kl_loss_type=custom`` on the command
     7: line routes the actor-side KL loss through this module.
     8: """
     9: 
    10: from typing import Optional
    11: 
    12: import torch
    13: import verl.trainer.ppo.core_algos as _core_algos
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom KL-divergence estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: def compute_custom_kl_penalty(
    21:     logprob: torch.Tensor,
    22:     ref_logprob: torch.Tensor,
    23: ) -> torch.Tensor:
    24:     """Per-token KL-divergence estimator used by the actor's KL loss.
    25: 
    26:     This function is called per micro-batch from ``dp_actor.py``'s
    27:     ``compute_kl_loss`` path (gated by ``actor.use_kl_loss=True``) after
    28:     the "custom" dispatch branch registered below.  It receives the
    29:     per-token log-probabilities of the current policy and the frozen
    30:     reference policy, and must return a per-token KL estimate of the
    31:     same shape.  The returned tensor is multiplied by
    32:     ``actor.kl_loss_coef`` and added to the policy-gradient loss.
    33: 
    34:     Reference forms implemented by verl's ``kl_penalty_forward``:
    35:         * k1 ("kl")         : logprob - ref_logprob            (unbiased, high variance)
    36:         * k2 ("mse")        : 0.5 * (logprob - ref_logprob)^2  (biased, low variance)
    37:         * k3 ("low_var_kl") : exp(ref - log) - (ref - log) - 1 (unbiased, low variance)
    38:         * abs               : |logprob - ref_logprob|          (robust to outliers)
    39: 
    40:     See J. Schulman, "Approximating KL divergence" (2020)
    41:     http://joschu.net/blog/kl-approx.html and DeepSeekMath
    42:     (arXiv:2402.03300) for the k3 estimator used in GRPO.
    43: 
    44:     Args:
    45:         logprob: (bs, response_length) log-probs under the current policy.
    46:         ref_logprob: (bs, response_length) log-probs under the frozen reference.
    47: 
    48:     Returns:
    49:         kl_estimate: (bs, response_length) per-token KL estimate.
    50:     """
    51:     # Default: k3 (low_var_kl) — the verl default.  Safe to run out of the box.
    52:     kl = ref_logprob - logprob
    53:     kl = torch.clamp(kl, min=-20, max=20)
    54:     ratio = torch.exp(kl)
    55:     kld = (ratio - kl - 1).contiguous()
    56:     return torch.clamp(kld, min=-10, max=10)
    57: 
    58: 
    59: # Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
    60: # Keep this at the bottom so the function definition above is in scope.
    61: _original_kl_penalty_forward = _core_algos.kl_penalty_forward
    62: 
    63: 
    64: def _patched_kl_penalty_forward(logprob, ref_logprob, kl_penalty):
    65:     """Dispatch ``kl_penalty=='custom'`` to ``compute_custom_kl_penalty``."""
    66:     if kl_penalty == "custom":
    67:         return compute_custom_kl_penalty(logprob, ref_logprob)
    68:     return _original_kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    69: 
    70: 
    71: _core_algos.kl_penalty_forward = _patched_kl_penalty_forward
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `k1` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_kl_penalty.py`:

```python
Lines 17–25:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom KL-divergence estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: def compute_custom_kl_penalty(
    21:     logprob: torch.Tensor,
    22:     ref_logprob: torch.Tensor,
    23: ) -> torch.Tensor:
    24:     """k1 estimator: naive unbiased KL = logprob - ref_logprob."""
    25:     return logprob - ref_logprob
    26: 
    27: 
    28: # Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
```

### `k2` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_kl_penalty.py`:

```python
Lines 17–25:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom KL-divergence estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: def compute_custom_kl_penalty(
    21:     logprob: torch.Tensor,
    22:     ref_logprob: torch.Tensor,
    23: ) -> torch.Tensor:
    24:     """k2 (mse) estimator: 0.5 * (logprob - ref_logprob) ** 2."""
    25:     return 0.5 * (logprob - ref_logprob).square()
    26: 
    27: 
    28: # Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
```

### `k3` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_kl_penalty.py`:

```python
Lines 17–29:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom KL-divergence estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: def compute_custom_kl_penalty(
    21:     logprob: torch.Tensor,
    22:     ref_logprob: torch.Tensor,
    23: ) -> torch.Tensor:
    24:     """k3 (low_var_kl): exp(r - l) - (r - l) - 1, unbiased, low variance."""
    25:     kl = ref_logprob - logprob
    26:     kl = torch.clamp(kl, min=-20, max=20)
    27:     ratio = torch.exp(kl)
    28:     kld = (ratio - kl - 1).contiguous()
    29:     return torch.clamp(kld, min=-10, max=10)
    30: 
    31: 
    32: # Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
```

### `abs` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_kl_penalty.py`:

```python
Lines 17–25:
    14: 
    15: # =====================================================================
    16: # EDITABLE: Implement your custom KL-divergence estimator below.
    17: # =====================================================================
    18: 
    19: 
    20: def compute_custom_kl_penalty(
    21:     logprob: torch.Tensor,
    22:     ref_logprob: torch.Tensor,
    23: ) -> torch.Tensor:
    24:     """abs estimator: |logprob - ref_logprob|."""
    25:     return (logprob - ref_logprob).abs()
    26: 
    27: 
    28: # Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
