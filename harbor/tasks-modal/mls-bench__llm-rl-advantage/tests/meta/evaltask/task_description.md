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
