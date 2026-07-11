"""Strong baseline for ape-meta-prompt (META-PROMPT design for reverse-mode instruction induction).

Reference: pasted into solution/meta_prompt.py via the edit op.
"""

import common  # noqa: F401


def meta_prompt(examples, ctx):
    # Strong: a structured REVERSE-MODE template ({demo} and {labels} are filled by
    # the harness) that steers the LM to emit exactly one concise task instruction.
    return ("I gave a friend an instruction and several examples. Read the "
            "input/output examples below and write ONE concise single-sentence "
            "instruction that maps each input to its output. The output is always "
            "exactly one of: {labels}. Output only the instruction, nothing else.\n\n"
            "{demo}\n\nThe instruction was:")
