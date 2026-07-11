"""Instruction-SEARCH surface (agent-editable) — automatic prompt optimization.

Frozen instruction LM, inference-only, ZERO-SHOT execution (no demonstrations in the
executed prompt — the demonstrations are used ONLY to PROPOSE instructions). You
design an APE optimizer that returns a single FINAL instruction string; the harness
scores it on a DISJOINT HELD-OUT TEST set. The base LM, the forced-choice executor,
the proposal pool, the dev/test split, and the label set are ALL FIXED — you control
ONLY how the instruction is proposed and selected.

Implement:

    def optimize(ctx) -> str:
        # ctx["executor"]  : Executor.
        #     .dev_accuracy(instruction, dev_rows) -> float
        #     .predict(instruction, rows) -> np.ndarray of label ids
        #     .label_logprob_matrix(instruction, rows) -> [n,n_class] log-probs
        # ctx["dataset"]   : the Dataset (labels, n_class, task, label_words()).
        # ctx["pool"]      : list[{text,label}] labeled examples to INDUCE from.
        # ctx["dev"]       : list[{text,label}] small dev set to SELECT on.
        # ctx["induce_instructions"](n) -> list[str]  LM-induced candidates
        #     (reverse-mode APE: the frozen LM writes candidate instructions from
        #      few-shot input/output examples drawn from the pool).
        # ctx["rng"]       : seeded random.Random.
        # return ONE instruction string (may be "" for the zero-instruction case).

The scored metric is TEST accuracy of your chosen instruction, so an instruction
overfit to the tiny dev set will NOT win — favor candidates that are robust on dev.

Ideas (headroom over the empty-instruction baseline):
  * PROPOSE: LM-induce candidates from the labeled pool (ctx["induce_instructions"]);
    optionally add a hand-written candidate; iterate (resample around the best).
  * SELECT: keep the candidate with the highest DEV accuracy (APE, Zhou et al. 2022).
  * A generic / empty instruction leaves the small LM near the class prior — a
    well-searched, task-specific instruction lifts zero-shot accuracy clearly.

Keep it deterministic. Both complete official dataset settings run serially on one GPU.
"""
from __future__ import annotations

import common  # noqa: F401


# ================================================================
# EDITABLE REGION — return your final instruction string below
# ================================================================
def optimize(ctx) -> str:
    # Default (weak): the empty / zero instruction — no task guidance at all.
    return ""
# ================================================================
# END EDITABLE REGION
# ================================================================
