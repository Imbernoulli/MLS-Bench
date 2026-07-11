"""Agent-editable decoding policy for cd-numeric-answer (GSM8K).

You control the INFERENCE-TIME constrained-decoding policy for a FROZEN small
instruction LM answering grade-school math word problems whose answer is an
integer. Return a `common.DecodeSpec` from `build_decoder`.

You decide:
  * the PROMPT wording,
  * WHAT to constrain: the entire output to just an integer (answer-only), OR
    let the model reason FIRST in free-form text and constrain ONLY the final
    integer (reason-then-constrain), via `preamble_regex` + `trigger`,
  * the ANSWER regex (must match the integer you want to score).

Correctness = the answer region is structurally VALID (matches your regex) AND
the extracted integer equals the gold integer. Emitting a valid but wrong
answer earns nothing. Over-constraining the whole output (forcing the very first
tokens to be a bare integer) suppresses chain-of-thought and typically lowers
accuracy on this reasoning-heavy task; letting the model reason first and
constraining only the final integer typically recovers it.

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
    # Default (WEAK): constrain the ENTIRE output to a bare integer immediately.
    # Structurally always valid, but the model must commit to a number before
    # doing any reasoning -> low accuracy on GSM8K.
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
