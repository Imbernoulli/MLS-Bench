"""Agent-editable decoding policy for cd-numeric-answer (GSM8K).

You control the INFERENCE-TIME constrained-decoding policy for a FROZEN small
instruction LM answering grade-school math word problems whose answer is an
integer. Return a `common.DecodeSpec` from `build_decoder`.

You decide:
  * the prompt wording,
  * whether the policy includes a free-form preamble,
  * the trigger and answer-region placement when a preamble is used,
  * the answer regex and explicit token budgets.
  * Every returned field is validated before decoding.

Correctness requires a structurally valid answer that matches a verifier-only
target. Structural validity alone does not earn credit. The benchmark does not
publish an ordering over prompt, preamble, trigger, or regex choices here.
Use the fixed live model and public source prompts to compare policies.
Runtime failures and non-finite model outputs abort verification.
No alternate decoder is substituted for an invalid surface.

`common.DecodeSpec` fields (see vendor/constrained-decoding-lab/common.py):
    DecodeSpec(prompt, answer_regex=..., choices=...,
               preamble_regex=..., trigger=..., max_answer_tokens=...)

Keep it deterministic (greedy is fixed by the harness).
"""
from __future__ import annotations

import common  # noqa: F401  (available on PYTHONPATH inside the harness)


# ================================================================
# EDITABLE REGION — return your DecodeSpec below
# ================================================================
def build_decoder(question: str, tok):
    # Native no-edit policy: apply an integer constraint immediately.
    # It is intentionally left runnable for no-op verifier evaluation.
    # Replace only this function when testing another decode policy.
    prompt = (
        "Answer the math problem with the final integer only.\n\n"
        f"Problem: {question}\nAnswer: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
