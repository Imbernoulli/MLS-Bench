"""REINFORCE++-baseline — rigorous codebase edit ops.

Port of verl's compute_reinforce_plus_plus_baseline_outcome_advantage
(vendor/external_packages/verl/verl/trainer/ppo/core_algos.py:422-473,
reference: https://arxiv.org/abs/2501.03262).

Per-prompt group-mean baseline broadcast to token level, then whitened
across the batch's valid response tokens (length-weighted).
"""

_FILE = "verl/verl/trainer/ppo/custom_advantage.py"

_RF_PP_BASELINE_ADVANTAGE = """\
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
    \"\"\"REINFORCE++-baseline: group-centered reward, token-level batch whitening.

    Subtract per-prompt group mean from each response's scalar reward,
    broadcast to token level, then masked-whiten over all valid response
    tokens in the batch (longer responses contribute more to the whitening
    statistics).
    \"\"\"
    response_length = token_level_rewards.shape[-1]
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
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 72,
        "content": _RF_PP_BASELINE_ADVANTAGE,
    },
]
