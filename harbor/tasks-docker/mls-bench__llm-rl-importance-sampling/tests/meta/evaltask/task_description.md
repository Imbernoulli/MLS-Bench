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
