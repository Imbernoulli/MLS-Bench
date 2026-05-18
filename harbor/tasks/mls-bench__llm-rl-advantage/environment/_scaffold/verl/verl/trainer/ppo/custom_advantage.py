# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Custom advantage estimator for verl PPO training."""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est

# =====================================================================
# EDITABLE: Implement your custom advantage estimator below.
# =====================================================================


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    ref_log_probs: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage estimates for policy optimization.

    This function is called by the verl training loop to compute per-token
    advantage and return estimates.  You may implement any strategy — e.g.
    group-normalized (GRPO), leave-one-out (RLOO), discounted cumulative
    returns (REINFORCE++), KL-penalized methods, or a novel combination.

    Args:
        token_level_rewards: (bs, response_length)
            Per-token rewards.  For outcome-based rewards the scalar reward
            is placed at the last valid token; use .sum(dim=-1) to recover
            per-sequence scores.
        response_mask: (bs, response_length)
            Binary mask indicating valid response tokens (1 = valid).
        index: (bs,) optional
            Group/prompt identifier for each sample.  Samples sharing the
            same index were generated from the same prompt and can be
            compared for group-based normalization.
        epsilon: Small constant to avoid division by zero.
        config: Algorithm configuration (AlgoConfig dataclass).
            Useful fields: config.gamma (discount factor, default 1.0),
            config.lam (GAE lambda), config.norm_adv_by_std_in_grpo, etc.
        old_log_probs: (bs, response_length) optional
            Log-probabilities of each token under the current rollout policy.
            Useful for entropy-based exploration bonuses, importance weighting,
            or probability-aware advantage shaping.
        ref_log_probs: (bs, response_length) optional
            Log-probabilities under the reference (frozen) policy.
            Useful for KL-penalized advantage: KL ≈ old_log_probs - ref_log_probs.

    Returns:
        advantages: (bs, response_length) — advantage estimates per token.
        returns:    (bs, response_length) — return estimates per token.

    Available utilities:
        verl_F.masked_whiten(values, mask)  — zero-mean unit-variance whitening
        verl_F.masked_mean(values, mask)    — masked mean
    """
    raise NotImplementedError(
        "Implement your custom advantage estimator here. "
        "See core_algos.py for reference implementations (GRPO, RLOO, REINFORCE++, etc.)."
    )
