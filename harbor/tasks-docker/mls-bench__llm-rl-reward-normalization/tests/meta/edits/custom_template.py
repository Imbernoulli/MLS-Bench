# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Custom reward-normalization for verl PPO training.

This function runs UPSTREAM of advantage estimation: it transforms the
raw scalar reward(s) assigned by the reward manager into normalized
per-token rewards, which then feed into advantage computation.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

# =====================================================================
# EDITABLE: Implement your custom reward normalization below.
# =====================================================================


def normalize_rewards(
    token_level_scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    config: Optional[object] = None,
    **kwargs,
) -> torch.Tensor:
    """Normalize per-response scalar rewards before advantage estimation.

    This is the UPSTREAM hook: it receives the raw reward tensor produced
    by the reward manager (outcome reward placed at the last valid token)
    and returns a transformed reward tensor of the same shape.  The result
    is written back into ``data.batch["token_level_scores"]`` and then
    consumed by the advantage estimator (GRPO / REINFORCE++ / GAE / ...).

    Args:
        token_level_scores: (bs, response_length)
            Raw per-token reward.  For outcome rewards only the last valid
            token is non-zero; ``token_level_scores.sum(dim=-1)`` recovers
            the per-sequence scalar.
        response_mask: (bs, response_length) binary mask (1 = valid token).
        index: (bs,) optional — group/prompt identifier.  Samples sharing
            the same index were generated from the same prompt (GRPO-style
            group of n rollouts).  Use ``defaultdict(list)`` to collect
            per-group statistics.
        epsilon: small constant to avoid division by zero.
        config: the full ``algorithm`` hydra config (DictConfig). Useful
            if you add per-strategy hyperparameters.

    Returns:
        token_level_scores: (bs, response_length) — normalized rewards,
        masked by ``response_mask``.  Must have the same shape as input.

    Available utilities:
        verl_F.masked_whiten(values, mask)  — zero-mean unit-variance
        verl_F.masked_mean(values, mask)    — masked mean
    """
    # Default: raw / identity — pass rewards through unchanged.  This
    # matches verl's current (unnormalized) reward handling; the GRPO
    # default advantage estimator already does its own group-std
    # normalization on top.  Replace this body to experiment with other
    # reward-space normalization strategies.
    #
    # To implement a new strategy, replace this body — e.g. batch-std
    # whitening, per-prompt group-std (GRPO), length-aware division,
    # percentile clipping, etc.  See task_description.md for references.
    with torch.no_grad():
        scores = token_level_scores * response_mask
        return scores
