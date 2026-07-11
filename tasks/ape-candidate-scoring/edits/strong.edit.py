"""Strong baseline: dev EXECUTION-ACCURACY estimator (Honovich et al. 2022).

Rank each candidate instruction by the frozen LM's calibrated execution accuracy on
the DEV set — the canonical APE selection metric. The highest-dev-accuracy candidate
is the one the small LM actually follows best, and it generalizes to the disjoint
held-out TEST set, cleanly beating the misleading distractors in the fixed pool.
Reference: vendor/prompt-optimization-lab/baselines/scoring_execacc.py
"""

_FILE = "prompt-optimization-lab/solution/scoring.py"

_CONTENT = '''def score_candidate(instruction, ctx) -> float:
    # Canonical APE score: dev execution-accuracy under the frozen calibrated LM.
    return ctx["executor"].dev_accuracy(instruction, ctx["dev"])'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 45, "content": _CONTENT},
]
