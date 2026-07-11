"""Agent-editable decoding policy for cd-choice-reasoning.

The editable surface controls how an optional preamble and the constrained answer region are combined. Evaluate alternatives without assuming an ordering.

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
def build_decoder(text: str, labels, tok):
    prompt = (
        "Classify the news topic. Reply with exactly one of: World, Sports, "
        "Business, Sci/Tech.\n\n"
        f"News: {text}\nTopic: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"World|Sports|Business|Sci/Tech",
        max_answer_tokens=8,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
