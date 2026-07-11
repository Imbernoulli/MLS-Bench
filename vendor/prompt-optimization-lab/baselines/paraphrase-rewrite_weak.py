"""Weak baseline for ape-paraphrase-rewrite (Instruction PARAPHRASE vs from-scratch rewrite).

Reference: pasted into solution/rewrite.py via the edit op.
"""

import common  # noqa: F401


def rewrite(seed, ctx):
    # Weak: echo the seed unchanged — no paraphrase, so dev-selection just keeps the
    # seed and you cannot improve over the plain fixed instruction.
    return [seed]
