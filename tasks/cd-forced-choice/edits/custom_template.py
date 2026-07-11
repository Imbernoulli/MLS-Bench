"""Agent-editable decoding policy for cd-forced-choice classification.

You control the INFERENCE-TIME decoding policy for a FROZEN small instruction LM
that must classify a short text into ONE label from a FIXED label set. Return a
`common.DecodeSpec` from `build_decoder`.

You decide:
  * the PROMPT wording (you may list the labels),
  * WHAT to constrain: constrain the output DIRECTLY to the fixed label set
    (`choices=labels` — guaranteed structurally valid, and the harness commits
    to the highest-probability label), OR let the model free-generate a single
    line and rely on the fixed extractor to map it to a label (`answer_regex`),
  * optionally a short reasoning preamble before the constrained label.

A sample is CORRECT only if the committed answer is EXACTLY one of the fixed
labels (VALID) AND equals the gold label. A decoder that always returns one
constant label is valid but only scores the majority-class rate.

`common.DecodeSpec` fields (see vendor/constrained-decoding-lab/common.py):
    DecodeSpec(prompt, answer_regex=..., choices=...,
               preamble_regex=..., trigger=..., max_answer_tokens=...)
"""
from __future__ import annotations

import common  # noqa: F401


# ================================================================
# EDITABLE REGION — return your DecodeSpec below
# ================================================================
def build_decoder(text: str, labels, tok):
    # Default (WEAK): free-generate one short line, no constraint. The model
    # often adds prose / punctuation so the answer fails the exact-label match
    # -> low validity, low accuracy.
    label_list = ", ".join(labels)
    prompt = (
        "You are a news topic classifier. Read the news snippet and reply with "
        f"exactly one topic from this list: {label_list}. "
        "World = international/politics, Sports = games and athletes, "
        "Business = companies/markets/economy, Sci/Tech = science and "
        "technology.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"[^\n]{1,40}",
        max_answer_tokens=16,
    )
# ================================================================
# END EDITABLE REGION
# ================================================================
