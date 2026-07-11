"""Search-STRATEGY surface (agent-editable) — budgeted candidate selection.

Frozen instruction LM, inference-only, zero-shot execution. The candidate pool and
the estimator (DEV execution-accuracy) are FIXED; candidates are ranked by their dev
accuracy under the frozen executor. But dev evaluations are EXPENSIVE and the harness
imposes a strict BUDGET on the number of (candidate, dev-example) executions. You
design ONLY the SEARCH/ALLOCATION over that budget and return the chosen instruction,
which is scored on a DISJOINT HELD-OUT TEST set.

Implement:

    def select(candidates, ctx) -> str:
        # candidates : list[str] the FIXED candidate instructions.
        # ctx["dev"]      : list[{text,label}] a LARGE dev set (sub-sample it!).
        # ctx["budget"]   : int, max (candidate, dev-example) executions allowed.
        # ctx["eval_on_dev"](instruction, dev_rows) -> float
        #     budget-guarded dev accuracy; each UNIQUE (instruction,dev-example)
        #     execution costs 1; the harness ABORTS you if you exceed the budget.
        # ctx["rng"] : seeded RNG.  ctx["dataset"] : the Dataset.
        # return ONE instruction string from `candidates`.

Because the budget is small relative to (#candidates * #dev), you CANNOT evaluate
every candidate on all of dev. Judging every candidate on a TINY dev slice overfits
dev noise and often picks a candidate that does not generalize to TEST; spending the
budget to estimate the top candidates on ENOUGH dev examples generalizes.

Ideas (headroom over the pick-first / tiny-slice baseline):
  * SUCCESSIVE HALVING / UCB: cheaply screen all candidates on a small dev slice,
    then spend the remaining budget re-evaluating the survivors on MORE dev examples.
  * Always give the finalists enough dev examples that the accuracy estimate is
    stable (low variance) before committing.
  * A degenerate strategy (return candidates[0], or rank all on 2 dev examples)
    picks a poorly-generalizing / misleading candidate.

Keep it deterministic. Both complete official dataset settings run serially on one GPU.
"""
from __future__ import annotations

import common  # noqa: F401


# ================================================================
# EDITABLE REGION — return the chosen instruction string below
# ================================================================
def select(candidates, ctx) -> str:
    # Default (weak): pick the first candidate without spending any dev budget.
    return candidates[0]
# ================================================================
# END EDITABLE REGION
# ================================================================
