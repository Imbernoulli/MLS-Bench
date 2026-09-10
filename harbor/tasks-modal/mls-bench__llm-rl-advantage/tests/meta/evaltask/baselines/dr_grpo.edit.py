"""Dr. GRPO baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with a Dr. GRPO advantage estimator.
Dr. GRPO omits the standard deviation normalization compared to GRPO,
using only mean subtraction for group-relative advantages.
"""

_FILE = "verl/verl/trainer/ppo/custom_advantage.py"

_DR_GRPO_ADVANTAGE = """\
# =====================================================================


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Dr. GRPO: GRPO without standard deviation normalization.

    Computes outcome-level advantages by subtracting the group mean reward,
    without dividing by the group standard deviation.
    \"\"\"
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 72,
        "content": _DR_GRPO_ADVANTAGE,
    },
]
