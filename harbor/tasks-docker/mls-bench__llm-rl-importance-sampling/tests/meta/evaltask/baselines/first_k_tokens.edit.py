"""First-K-tokens (truncated IS) baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with a policy loss that applies the normal
per-token importance ratio for response positions t < K (K=64), but for
positions t >= K replaces the ratio with its DETACHED value. This
prevents gradients from flowing through the log-prob of late tokens
via the IS ratio, while still letting those tokens contribute to the
REINFORCE-like gradient through the clipped-loss `log_prob` factor
implicit in autograd (the loss is still -advantages * ratio, so
detaching ratio kills the gradient contribution for those tokens).

Variance / bias tradeoff: late tokens in long responses have the
largest log_prob - old_log_prob drift (compounding across the sequence)
and therefore the noisiest ratios. Freezing the ratio for t >= K
reduces gradient variance substantially at the cost of a small bias
(the policy gradient is no longer an unbiased IS estimator on those
positions). DAPO and related work explore similar truncation / decoupled
clipping schemes (https://arxiv.org/abs/2503.14476).
"""

_FILE = "verl/verl/trainer/ppo/custom_policy_loss.py"

_FIRST_K_LOSS = """\
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
    \"\"\"First-K truncated IS: per-token ratio for t<K, detached for t>=K.\"\"\"
    assert config is not None
    K = 64  # prefix length with live IS gradient
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Build prefix mask: 1 for the first K response positions, 0 afterwards.
    T = ratio.shape[-1]
    positions = torch.arange(T, device=ratio.device).unsqueeze(0)  # (1, T)
    prefix_mask = (positions < K).to(ratio.dtype)                  # (1, T)
    # Detach ratio beyond prefix: ratio_eff = ratio*prefix + detach(ratio)*(1-prefix)
    ratio_eff = ratio * prefix_mask + ratio.detach() * (1.0 - prefix_mask)

    pg_losses1 = -advantages * ratio_eff
    pg_losses2 = -advantages * torch.clamp(ratio_eff, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
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
        "content": _FIRST_K_LOSS,
    },
]
