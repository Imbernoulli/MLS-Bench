#!/usr/bin/env python3
"""ape-search-strategy harness (fixed pipeline).

Frozen instruction LM, inference-only, zero-shot execution. The candidate POOL and
the ESTIMATOR are FIXED here: candidates are ranked by their DEV execution-accuracy,
computed by the FIXED executor. But the DEV set is LARGE and evaluating every
candidate on the whole dev set is expensive, so the harness imposes a strict BUDGET
on the number of (candidate, dev-example) executions.

The agent designs ONLY the SEARCH/ALLOCATION `select(candidates, ctx)`: given the
candidate list and a fixed budget of dev executions, decide how many dev examples to
spend on which candidates and return the single chosen instruction. A naive strategy
that judges every candidate on a TINY dev slice overfits dev noise and often picks a
candidate that does not generalize; a strategy that spends the budget to estimate the
top candidates on ENOUGH dev examples (e.g. successive halving / UCB) picks the
candidate that generalizes to the DISJOINT HELD-OUT TEST set. The chosen instruction
is scored on TEST.

The executor COUNTS executions; the harness aborts the agent if it exceeds the
budget, so a "just evaluate everything on all of dev" cheat is impossible.

Only the returned instruction is evaluated on the official evaluation split. The
terminal ``APE_RESULT`` line proves the complete inventory and single test pass.
"""
from __future__ import annotations

import argparse
import time

import common


class BudgetExceeded(SystemExit):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--dataset", required=True)      # agnews | sst2
    ap.add_argument("--budget", type=int, default=200)  # dev executions allowed
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()

    ds = common.load_dataset(args.dataset)
    executor = common.Executor(ds, seed=args.seed)
    select = common.load_surface(args.solution, "select")

    candidates = common.build_candidate_pool(executor, ds, seed=args.seed)
    # Reset the counter so the induction cost of building the pool does not count
    # against the agent's search budget (the pool is FIXED infrastructure).
    executor.n_exec_calls = 0
    budget = int(args.budget)

    def eval_on_dev(instruction: str, dev_rows):
        """Budget-guarded dev evaluation the agent must use to score candidates.
        Each UNIQUE (instruction, dev-example) execution costs 1; the harness
        aborts if the agent exceeds the budget."""
        before = executor.n_exec_calls
        preds = executor.predict(instruction, dev_rows)
        if executor.n_exec_calls > budget:
            raise BudgetExceeded(
                f"dev-execution budget {budget} exceeded "
                f"(used {executor.n_exec_calls})")
        acc = common.accuracy(preds, dev_rows)
        _ = before
        return acc

    import random as _r
    ctx = {
        "dataset": ds,
        "dev": ds.dev,                 # LARGE dev set (must be sub-sampled)
        "budget": budget,
        "eval_on_dev": eval_on_dev,    # budget-guarded dev accuracy
        "rng": _r.Random(args.seed),
        "n_candidates": len(candidates),
    }

    chosen = select(candidates, ctx)
    if not isinstance(chosen, str):
        raise SystemExit("select(candidates, ctx) must return an instruction STRING")

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
