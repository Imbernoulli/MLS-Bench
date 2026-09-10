"""group_std baseline — GRPO-style per-prompt group normalization,
applied UPSTREAM of the advantage estimator.

For each prompt group (samples sharing the same ``index``) we subtract
the group mean and divide by the group std (+eps), then re-place the
normalized scalar at the last valid token of each response.

This mirrors verl's ``compute_grpo_outcome_advantage`` normalization
(vendor/external_packages/verl/verl/trainer/ppo/core_algos.py) but
moves it to the reward stage.  Because the advantage estimator chosen
in train.sh is still GRPO, it will re-center+re-divide on top of this
(idempotent-up-to-epsilon), which provides a useful "double-normalize"
reference for the agent.

References:
- Shao et al., DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- verl GRPO normalization: core_algos.compute_grpo_outcome_advantage.
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
    \"\"\"group_std: per-prompt group mean + std normalization (GRPO-style).\"\"\"
    with torch.no_grad():
        bsz, seq_len = token_level_scores.shape
        scores = token_level_scores.sum(dim=-1)  # (bs,)

        if index is None:
            # Fallback to batch-level normalization if no grouping info.
            mean = scores.mean()
            std = scores.std(unbiased=False)
            scores = (scores - mean) / (std + epsilon)
        else:
            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx, vs in id2score.items():
                if len(vs) == 1:
                    id2mean[idx] = torch.tensor(0.0, device=scores.device)
                    id2std[idx] = torch.tensor(1.0, device=scores.device)
                else:
                    stacked = torch.stack(vs)
                    id2mean[idx] = stacked.mean()
                    id2std[idx] = stacked.std(unbiased=False)
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # Place the normalized scalar back at the last valid token of each
        # response so the outcome-reward semantics are preserved.
        out = torch.zeros_like(token_level_scores)
        last_idx = response_mask.long().sum(dim=-1) - 1  # (bs,)
        last_idx = last_idx.clamp(min=0)
        out[torch.arange(bsz, device=out.device), last_idx] = scores
        return out * response_mask
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
