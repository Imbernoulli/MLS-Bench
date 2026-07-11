"""Agent-editable decoding policy for cd-numeric-json.

The editable surface defines a structured numeric answer grammar and its placement in the decode policy.

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
        "Answer the math problem. Output ONLY a JSON object with the integer "
        "answer and nothing else.\n\n"
        f"Problem: {question}\nJSON: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r'\{"answer":-?[0-9]{1,7}\}',
        max_answer_tokens=16,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
