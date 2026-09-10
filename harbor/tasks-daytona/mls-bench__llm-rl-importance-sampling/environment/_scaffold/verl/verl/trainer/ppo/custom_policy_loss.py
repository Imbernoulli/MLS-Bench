# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Custom policy loss / importance-sampling strategy for verl PPO training."""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss

# =====================================================================
# EDITABLE: Implement your custom importance-sampling policy loss below.
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
    """Compute the clipped policy objective for LLM online RL.

    This function is called by the verl training loop.  The core design
    axis is *importance-sampling granularity*: how the ratio
        r = exp(log_prob - old_log_prob)
    is formed and clipped (per-token, per-sequence, truncated to a
    prefix, etc.).  See GSPO (Zheng et al., 2025, arXiv:2507.18071),
    DAPO (arXiv:2503.14476), and CISPO (MiniMax M1, arXiv:2506.13585)
    for references.

    Args:
        old_log_prob: (bs, response_length)
            Log-probabilities of each token under the old (rollout) policy.
        log_prob: (bs, response_length)
            Log-probabilities of each token under the current policy.
        advantages: (bs, response_length)
            Per-token advantage estimates.
        response_mask: (bs, response_length)
            Binary mask (1 = valid response token).
        loss_agg_mode: Aggregation mode forwarded to ``agg_loss``.
            Typical values: "token-mean", "seq-mean-token-mean".
        config: ``ActorConfig`` with fields such as ``clip_ratio``,
            ``clip_ratio_low``, ``clip_ratio_high``, and
            ``global_batch_info`` (passed as kwargs to ``agg_loss``).
            ``config.get("name", default)`` is supported for optional
            fields like ``clip_ratio_c``.
        rollout_is_weights: Optional per-token rollout-correction weights.

    Returns:
        pg_loss: scalar policy-gradient loss tensor.
        metrics: dict with at least ``actor/pg_clipfrac`` and
            ``actor/ppo_kl`` as Python floats.

    Typical call to aggregate:
        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
    """
    raise NotImplementedError(
        "Implement your custom importance-sampling policy loss here. "
        "See core_algos.py for reference (compute_policy_loss_vanilla / gspo)."
    )
