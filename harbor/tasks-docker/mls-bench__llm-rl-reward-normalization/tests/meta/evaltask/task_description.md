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
