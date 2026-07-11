"""Weak baseline: RANDOM / constant estimator — no selection signal.

Return a fixed constant (0.0) for every candidate. The harness's greedy argmax then
selects an arbitrary candidate (the first, given ties), which — with misleading
distractors in the fixed pool — often generalizes poorly, scoring near the class
prior. Reference: vendor/prompt-optimization-lab/solution/scoring.py (default).
"""


def score_candidate(instruction, ctx) -> float:
    return 0.0
