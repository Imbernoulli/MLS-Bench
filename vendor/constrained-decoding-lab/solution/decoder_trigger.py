"""Agent-editable decoding policy for cd-numeric-trigger.

Trigger-delimiter design is the editable axis. Compare prompt-aligned triggers under the same fixed decode and evaluation budget.

`common.DecodeSpec` fields (see vendor/constrained-decoding-lab/common.py):
    DecodeSpec(prompt, answer_regex=..., choices=...,
               preamble_regex=..., trigger=..., max_answer_tokens=...,
               max_free_tokens=...)
"""
from __future__ import annotations

import common  # noqa: F401  (available on PYTHONPATH inside the harness)


# ================================================================
# EDITABLE REGION — return your DecodeSpec below
# ================================================================
def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step, then give the final integer.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger=" ",
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
        max_free_tokens=256,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
