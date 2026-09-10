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
- **Policy**: Qwen2.5-0.5B (full-parameter), verl framework, GRPO advantage estimator.
- **Training set**: simpleRL-Zoo MATH level 3–5 (Qwen split) plus a 5K subset of DeepMath.
- **Hyperparameters**: 100 steps, 16 rollout samples per prompt, batch size 128, `actor.use_kl_loss=True`, `actor.kl_loss_coef=0.001`, `actor.kl_loss_type=custom`.
- Reward manager, advantage estimator, rollout, model, optimizer, and evaluation are fixed.

## Evaluation
Math-reasoning accuracy (`mean@1`) on **GSM8K**, **MATH-500**, and **AMC 23**; primary score is the mean across the three.
