"""outcome_only baseline — raw reward pass-through.

No normalization is applied: the per-response scalar reward stays at the
last valid token exactly as the reward manager emitted it, and the GRPO
advantage estimator downstream does all of the normalization.  This is
the "naive" / verl-default state and the head-to-head reference for all
other normalization strategies in this task.

References:
- Shao et al., DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- verl default reward handling: vendor/external_packages/verl/verl/trainer/ppo/ray_trainer.py
  (line 1443: ``batch.batch["token_level_scores"] = reward_tensor`` — the reward
  enters ``compute_advantage`` unmodified).
"""

_FILE = "verl/verl/trainer/ppo/custom_reward_normalization.py"

_BODY = """\
# =====================================================================


def normalize_rewards(
    token_level_scores,
    response_mask,
    index=None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    \"\"\"outcome_only: no reward-space normalization.

    Pass the raw (bs, response_length) reward tensor straight through.
    The outcome reward is left at the last valid token and whatever
    downstream advantage estimator is configured (GRPO by default)
    applies its own normalization.
    \"\"\"
    with torch.no_grad():
        return token_level_scores * response_mask
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 72,
        "content": _BODY,
    },
]
