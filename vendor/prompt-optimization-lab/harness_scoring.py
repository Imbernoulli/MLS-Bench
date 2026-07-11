#!/usr/bin/env python3
"""ape-candidate-scoring harness (fixed pipeline).

Frozen instruction LM, inference-only, zero-shot execution. The candidate POOL of
instructions and the SEARCH RULE are FIXED here: the harness gives the agent a fixed
list of candidate instructions and (after the agent scores them) picks the
single-highest-scoring candidate (greedy argmax over the agent's scores), then
evaluates THAT candidate on the DISJOINT HELD-OUT TEST set.

The agent designs ONLY the ESTIMATOR `score_candidate(instruction, ctx)` — the
number used to RANK candidates. A good estimator (dev execution-accuracy, or dev
answer log-likelihood) surfaces the candidate that truly generalizes; a
random/constant estimator picks a poor candidate and scores near the class prior.
Because ranking uses only DEV but the metric is on the disjoint TEST set, an
estimator that overfits dev noise does not win.

Only the argmax-selected instruction is evaluated on the official evaluation split.
The terminal ``APE_RESULT`` line proves the complete inventory and single test pass.
"""
from __future__ import annotations

import argparse
import math
import time

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--dataset", required=True)      # agnews | sst2
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()

    ds = common.load_dataset(args.dataset)
    executor = common.Executor(ds, seed=args.seed)
    score_candidate = common.load_surface(args.solution, "score_candidate")

    # FIXED candidate pool: LM-induced from the labeled pool + a couple of fixed
    # anchors (a strong hand-written one and a deliberately-misleading one), so a
    # discriminating estimator has a clear best to find and a clear worst to avoid.
    candidates = common.build_candidate_pool(executor, ds, seed=args.seed)

    import random as _r
    ctx = {
        "executor": executor,
        "dataset": ds,
        "dev": ds.dev,
        "pool": ds.pool,
        "rng": _r.Random(args.seed),
    }

    scores = []
    for instr in candidates:
        s = score_candidate(instr, ctx)
        scores.append(float(s))
    if len(scores) != len(candidates):
        raise SystemExit("score_candidate must return one score per candidate")
    if not all(math.isfinite(score) for score in scores):
        raise SystemExit("score_candidate returned a non-finite rank score")

    # FIXED search rule: greedy argmax of the agent's estimator.
    best_i = max(range(len(candidates)), key=lambda i: scores[i])
    chosen = candidates[best_i]

    print(f"APE_RANK selected_index={best_i} rank_score={scores[best_i]:.9f}",
          flush=True)
    common.emit_selected_metrics(
        executor,
        ds,
        chosen,
        t0,
        n_candidates=len(candidates),
        dev_exec_calls=executor.n_exec_calls,
    )


if __name__ == "__main__":
    main()
