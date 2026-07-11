"""Agent-editable prompt policy for error-driven code repair.

The verifier owns all generation calls. This surface returns only the prompt
used for one fixed greedy repair generation after a PROVIDED-test failure.
"""
from __future__ import annotations


def build_repair_prompt(problem, program, error, round_index):
    """Build the prompt for one of at most two compute-matched repair rounds."""
    return (
        "Review the candidate function and return a corrected complete function "
        "inside one Python code block.\n\n"
        f"Task:\n{problem['prompt']}\n\nCandidate:\n```python\n{program}\n```"
    )
