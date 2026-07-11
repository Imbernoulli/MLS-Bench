"""Weak baseline: RANDOM / constant estimator — no selection signal.

Return a fixed constant for every candidate. The harness's greedy argmax then selects
an arbitrary candidate (the first, given ties), which — with the misleading
distractors in the FIXED pool — generalizes poorly, scoring near the class prior.
Reference: vendor/prompt-optimization-lab/solution/scoring.py (default).
"""

_FILE = "prompt-optimization-lab/solution/scoring.py"

_CONTENT = '''def score_candidate(instruction, ctx) -> float:
    # Constant score: ranks candidates arbitrarily (no signal).
    return 0.0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 45, "content": _CONTENT},
]
