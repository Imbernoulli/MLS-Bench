"""k1 (naive, "kl") baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with the k1 KL-divergence estimator:
    kl = logprob - ref_logprob.
This is the unbiased, high-variance Monte Carlo estimator.  It matches
verl's ``kl_penalty_forward`` branch for kl_penalty in ("kl", "k1").

Reference: J. Schulman, "Approximating KL divergence" (2020).
http://joschu.net/blog/kl-approx.html
"""

_FILE = "verl/verl/trainer/ppo/custom_kl_penalty.py"

_K1_BODY = """\
# =====================================================================


def compute_custom_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    \"\"\"k1 estimator: naive unbiased KL = logprob - ref_logprob.\"\"\"
    return logprob - ref_logprob
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 56,
        "content": _K1_BODY,
    },
]
