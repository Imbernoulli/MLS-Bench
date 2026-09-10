"""k2 ("mse") baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with the k2 KL-divergence estimator:
    kl = 0.5 * (logprob - ref_logprob) ** 2.
This is Schulman's low-variance biased estimator.  It matches verl's
``kl_penalty_forward`` branch for kl_penalty in ("mse", "k2").

Reference: J. Schulman, "Approximating KL divergence" (2020).
http://joschu.net/blog/kl-approx.html
"""

_FILE = "verl/verl/trainer/ppo/custom_kl_penalty.py"

_K2_BODY = """\
# =====================================================================


def compute_custom_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    \"\"\"k2 (mse) estimator: 0.5 * (logprob - ref_logprob) ** 2.\"\"\"
    return 0.5 * (logprob - ref_logprob).square()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 56,
        "content": _K2_BODY,
    },
]
