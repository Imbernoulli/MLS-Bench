"""Agent-editable decoding policy for cd-forced-choice classification.

You control the INFERENCE-TIME decoding policy for a FROZEN small instruction LM
that must classify a short text into ONE label from a FIXED label set. Return a
`common.DecodeSpec` from `build_decoder`.

You decide:
  * the prompt wording,
  * whether the answer region uses literal choices or a regular expression,
  * the corresponding answer-region token budget,
  * whether a free-form preamble is part of the policy,
  * and any trigger required by that preamble.

The verifier checks structural validity and correctness against private targets.
The solution interface does not expose target labels or baseline ordering.
Invalid surface values fail instead of selecting a different decoder.

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
    # Native policy: free-generate one short line under an explicit regex.
    # This remains a valid no-edit workspace for verifier execution.
    # Replace only this declared function when exploring another policy.
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
