"""abs baseline — rigorous codebase edit ops.

Replaces the EDITABLE region with the absolute-difference KL estimator:
    kl = |logprob - ref_logprob|.
This matches verl's ``kl_penalty_forward`` branch for kl_penalty == "abs".
It is a robust-to-outliers alternative to the squared k2 estimator.
"""

_FILE = "verl/verl/trainer/ppo/custom_kl_penalty.py"

_ABS_BODY = """\
# =====================================================================


def compute_custom_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    \"\"\"abs estimator: |logprob - ref_logprob|.\"\"\"
    return (logprob - ref_logprob).abs()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 56,
        "content": _ABS_BODY,
    },
]
