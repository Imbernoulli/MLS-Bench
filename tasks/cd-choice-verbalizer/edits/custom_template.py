"""Agent-editable decoding policy for cd-choice-verbalizer.

Label VERBALIZER / choice-set design under an EXACT-match evaluator: the strings in `choices` must coincide with the gold labels; paraphrases are committed by the model but rejected by the exact-label check (valid=0).

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
        "You are a news topic classifier. Pick exactly one topic from: "
        "World News, Sports News, Business News, Science/Tech.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=["World News", "Sports News", "Business News", "Science/Tech"],
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
