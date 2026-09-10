"""batch_std baseline — batch-level reward whitening.

Subtract the batch mean of per-response scalars and divide by the batch
std + eps.  Ignores per-prompt grouping — every sample in the batch is
normalized against the same (batch mean, batch std) pair.  This is a common
RLHF-style whitening strategy.

References:
- Ouyang et al., InstructGPT, 2022: https://arxiv.org/abs/2203.02155
  (value-function whitening is the direct analogue; Stiennon et al.
  "Learning to summarize from human feedback" NeurIPS 2020 whiten the
  reward itself, same idea.)
- TRL PPO trainer exposes a ``whiten_rewards`` flag:
  https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_config.py
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
    \"\"\"batch_std: subtract batch mean, divide by batch std + eps.\"\"\"
    with torch.no_grad():
        bsz, seq_len = token_level_scores.shape
        scores = token_level_scores.sum(dim=-1)  # (bs,)

        if bsz <= 1:
            # Degenerate case — no normalization possible.
            mean = torch.tensor(0.0, device=scores.device)
            std = torch.tensor(1.0, device=scores.device)
        else:
            mean = scores.mean()
            std = scores.std(unbiased=False)

        scores = (scores - mean) / (std + epsilon)

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
