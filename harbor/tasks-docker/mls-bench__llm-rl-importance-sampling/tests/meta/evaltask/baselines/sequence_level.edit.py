"""Sequence-level IS baseline (GSPO-style) — rigorous codebase edit ops.

Replaces the EDITABLE region with a policy loss that uses a SINGLE
importance ratio per sequence:
    s_i = exp( mean_t (log_prob_new - log_prob_old) )   (masked mean)
broadcast to every token, then clip + PPO loss. The straight-through
trick `log_prob - log_prob.detach() + detach(log_seq_ratio)` keeps the
per-token log_prob gradient while using the sequence-level ratio value.

Reference: GSPO, Zheng et al. 2025, https://arxiv.org/abs/2507.18071
"""

_FILE = "verl/verl/trainer/ppo/custom_policy_loss.py"

_SEQ_LEVEL_LOSS = """\
# =====================================================================


@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    \"\"\"Sequence-level IS (GSPO): one scalar ratio per sequence.\"\"\"
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    neg_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # straight-through: keep per-token log_prob gradient, ratio value is per-sequence
    log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    seq_ratio = torch.exp(log_seq_ratio)

    pg_losses1 = -advantages * seq_ratio
    pg_losses2 = -advantages * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # GSPO aggregates at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 72,
        "content": _SEQ_LEVEL_LOSS,
    },
]
