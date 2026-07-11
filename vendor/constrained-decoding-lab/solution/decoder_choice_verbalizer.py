"""Agent-editable decoding policy for cd-choice-verbalizer.

The editable surface controls the literal choice set, its one-to-one canonical
label mapping, and the prompt used by a verifier-target evaluator.

`common.DecodeSpec` fields (see vendor/constrained-decoding-lab/common.py):
    DecodeSpec(prompt, answer_regex=..., choices=..., choice_labels=...,
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
        "International, Athletics, Markets, Technology.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=["International", "Athletics", "Markets", "Technology"],
        choice_labels=list(labels),
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
