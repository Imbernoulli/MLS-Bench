"""k3 ("low_var_kl") baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with the k3 KL-divergence estimator:
    log_ratio = ref_logprob - logprob
    kl = exp(log_ratio) - log_ratio - 1
This is Schulman's unbiased low-variance estimator and verl's default
``kl_loss_type`` for GRPO.  It matches verl's ``kl_penalty_forward``
branch for kl_penalty in ("low_var_kl", "k3") including the numerical
clamps.

Reference: J. Schulman, "Approximating KL divergence" (2020).
http://joschu.net/blog/kl-approx.html and DeepSeekMath (arXiv:2402.03300).
"""

_FILE = "verl/verl/trainer/ppo/custom_kl_penalty.py"

_K3_BODY = """\
# =====================================================================


def compute_custom_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    \"\"\"k3 (low_var_kl): exp(r - l) - (r - l) - 1, unbiased, low variance.\"\"\"
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 56,
        "content": _K3_BODY,
    },
]
