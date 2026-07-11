"""Weak baseline for ape-meta-prompt (META-PROMPT design for reverse-mode instruction induction).

Reference: pasted into solution/meta_prompt.py via the edit op.
"""

import common  # noqa: F401


def meta_prompt(examples, ctx):
    # Weak: a vague prompt — the LM rambles off-task instead of emitting a clean task
    # instruction, so induction yields noise and dev-selection has nothing good.
    return "Say something about these examples."
