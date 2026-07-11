"""Trusted full-split anchor surface for cd-numeric-answer."""
from __future__ import annotations

import common


def build_decoder(question: str, tok):
    prompt = (
        "Answer the math problem with the final integer only.\n\n"
        f"Problem: {question}\nAnswer: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
    )
