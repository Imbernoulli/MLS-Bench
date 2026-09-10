# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Custom KL-divergence estimator for verl PPO training (actor KL loss).

Registers a "custom" branch on ``core_algos.kl_penalty_forward`` so that
passing ``actor_rollout_ref.actor.kl_loss_type=custom`` on the command
line routes the actor-side KL loss through this module.
"""

from typing import Optional

import torch
import verl.trainer.ppo.core_algos as _core_algos

# =====================================================================
# EDITABLE: Implement your custom KL-divergence estimator below.
# =====================================================================


def compute_custom_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    """Per-token KL-divergence estimator used by the actor's KL loss.

    This function is called per micro-batch from ``dp_actor.py``'s
    ``compute_kl_loss`` path (gated by ``actor.use_kl_loss=True``) after
    the "custom" dispatch branch registered below.  It receives the
    per-token log-probabilities of the current policy and the frozen
    reference policy, and must return a per-token KL estimate of the
    same shape.  The returned tensor is multiplied by
    ``actor.kl_loss_coef`` and added to the policy-gradient loss.

    Reference forms implemented by verl's ``kl_penalty_forward``:
        * k1 ("kl")         : logprob - ref_logprob            (unbiased, high variance)
        * k2 ("mse")        : 0.5 * (logprob - ref_logprob)^2  (biased, low variance)
        * k3 ("low_var_kl") : exp(ref - log) - (ref - log) - 1 (unbiased, low variance)
        * abs               : |logprob - ref_logprob|          (robust to outliers)

    See J. Schulman, "Approximating KL divergence" (2020)
    http://joschu.net/blog/kl-approx.html and DeepSeekMath
    (arXiv:2402.03300) for the k3 estimator used in GRPO.

    Args:
        logprob: (bs, response_length) log-probs under the current policy.
        ref_logprob: (bs, response_length) log-probs under the frozen reference.

    Returns:
        kl_estimate: (bs, response_length) per-token KL estimate.
    """
    # Default: k3 (low_var_kl) — the verl default.  Safe to run out of the box.
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)


# Wiring below: register "custom" branch on core_algos.kl_penalty_forward.
# Keep this at the bottom so the function definition above is in scope.
_original_kl_penalty_forward = _core_algos.kl_penalty_forward


def _patched_kl_penalty_forward(logprob, ref_logprob, kl_penalty):
    """Dispatch ``kl_penalty=='custom'`` to ``compute_custom_kl_penalty``."""
    if kl_penalty == "custom":
        return compute_custom_kl_penalty(logprob, ref_logprob)
    return _original_kl_penalty_forward(logprob, ref_logprob, kl_penalty)


_core_algos.kl_penalty_forward = _patched_kl_penalty_forward
