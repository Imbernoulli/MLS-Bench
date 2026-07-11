"""Candidate-SCORING surface (agent-editable) — the APE selection ESTIMATOR.

Frozen instruction LM, inference-only, zero-shot execution. The candidate pool of
instructions and the SEARCH RULE are FIXED (the harness gives you a fixed list of
candidates and picks the single-highest-scoring one, then scores it on a DISJOINT
HELD-OUT TEST set). You design ONLY the ESTIMATOR that RANKS candidates.

Implement:

    def score_candidate(instruction, ctx) -> float:
        # instruction : str, one candidate instruction to score (rank by this).
        # ctx["executor"] : Executor.
        #     .dev_accuracy(instruction, dev_rows) -> float
        #     .predict(instruction, rows) -> label ids
        #     .label_logprob_matrix(instruction, rows) -> [n,n_class] log-probs
        # ctx["dev"]     : list[{text,label}] the dev set to score on.
        # ctx["dataset"] : the Dataset.  ctx["rng"] : seeded RNG.
        # return a float; HIGHER means "more likely the best instruction".

The harness picks argmax over your scores, then measures TEST accuracy. A good
estimator surfaces the candidate that truly generalizes; a random/constant
estimator picks a poor (even misleading) candidate and scores near the class prior.
Because ranking uses only DEV but the score is on the disjoint TEST set, an estimator
that chases dev noise does not win.

Ideas (headroom over the random estimator):
  * EXECUTION ACCURACY on dev (Honovich et al. 2022): return the candidate's dev
    accuracy under the frozen LM — the canonical APE selection metric.
  * ANSWER LOG-LIKELIHOOD: mean gold-label log-prob on dev (a smoother estimator).
  * A constant/random score ranks candidates arbitrarily and often selects a
    deliberately-misleading distractor in the pool.

Keep it deterministic. Both complete official dataset settings run serially on one GPU.
"""
from __future__ import annotations

import common  # noqa: F401


# ================================================================
# EDITABLE REGION — return a scalar rank-score for `instruction`
# ================================================================
def score_candidate(instruction, ctx) -> float:
    # Default (weak): constant score — ranks candidates arbitrarily (no signal).
    return 0.0
# ================================================================
# END EDITABLE REGION
# ================================================================
