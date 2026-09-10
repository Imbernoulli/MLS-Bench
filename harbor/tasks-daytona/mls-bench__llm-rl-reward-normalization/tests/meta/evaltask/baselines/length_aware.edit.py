"""length_aware baseline — DAPO-style length-bias correction.

Divide each response's scalar reward by √(response_length) before
placing it back at the last valid token.  DAPO observed that longer
responses accumulate more per-token gradient signal under group-mean
normalization, biasing the policy toward verbosity; dividing the
reward by a function of length counter-balances that effect.

We use √T (as opposed to T) because it matches the per-token gradient
scaling implied by GRPO's group-mean baseline: ignoring std, the
per-token advantage is broadcast uniformly across T tokens, so the
total contribution scales as r · T / √T = r · √T after normalization;
pre-dividing by √T makes the per-sequence contribution length-invariant.

References:
- Liu et al., DAPO (Decoupled Clip and Dynamic Sampling Policy
  Optimization), 2025: https://arxiv.org/abs/2503.14476
- Dr. GRPO length-bias discussion: https://arxiv.org/abs/2503.20783
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
    \"\"\"length_aware: divide scalar reward by sqrt(response_length).\"\"\"
    with torch.no_grad():
        bsz, seq_len = token_level_scores.shape
        scores = token_level_scores.sum(dim=-1)  # (bs,)

        lengths = response_mask.sum(dim=-1).to(scores.dtype)  # (bs,)
        denom = torch.sqrt(lengths.clamp(min=1.0)) + epsilon
        scores = scores / denom

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
