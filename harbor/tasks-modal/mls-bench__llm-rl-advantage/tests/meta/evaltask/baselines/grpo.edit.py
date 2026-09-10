"""GRPO baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with a GRPO (Group Relative Policy Optimization)
advantage estimator, adapted from core_algos.py.
"""

_FILE = "verl/verl/trainer/ppo/custom_advantage.py"

_GRPO_ADVANTAGE = """\
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
    \"\"\"GRPO: Group Relative Policy Optimization advantage estimator.

    Computes outcome-level advantages by normalizing rewards within each
    prompt group by the group mean and standard deviation.
    \"\"\"
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    norm_adv_by_std = True
    if config is not None:
        norm_adv_by_std = getattr(config, "norm_adv_by_std_in_grpo", True)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
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
        "content": _GRPO_ADVANTAGE,
    },
]
