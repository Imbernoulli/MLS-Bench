"""Agent-editable decoding policy for cd-numeric-repair.

The answer-region token budget is the editable axis. Prompt, trigger, numeric
language, and free-reasoning budget stay fixed so the experiment isolates
answer-region dead ends.

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
        "Solve the math problem step by step. After your reasoning, write "
        "'#### ' followed by the final integer answer.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="#### ",
        answer_regex=r"[ ]?-?[0-9]{1,10}",
        max_answer_tokens=1,
        max_free_tokens=256,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
