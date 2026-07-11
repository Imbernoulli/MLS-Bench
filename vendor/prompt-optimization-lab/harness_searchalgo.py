#!/usr/bin/env python3
"""ape generative-search harness (fixed pipeline) — SEARCH ALGORITHM and shared
PROPOSAL/EVAL BUDGET-ALLOCATION surfaces.

Frozen instruction LM, inference-only, zero-shot calibrated execution. There is NO
fixed candidate pool here: the agent GENERATES candidates (LM induction / paraphrase)
and searches under a budget, returning ONE final instruction scored on the DISJOINT
HELD-OUT TEST set (fixed "N/A" calibration).

  --mode search  (ape-search-algorithm)
      search(ctx) -> str : the budget covers DEV evaluations only. ctx tools:
        induce(n)      -> n LM-induced candidates from the labeled pool (proposal)
        paraphrase(s,n)-> n meaning-preserving rewrites of s (refinement)
        eval_on_dev(instr, rows) -> budget-guarded dev accuracy (charges budget)
      A degenerate search proposes ONE candidate and returns it with NO dev check; a
      beam/iterative search proposes a few, dev-scores them, keeps the best, refines
      (paraphrase) and re-scores — surfacing an instruction that generalizes.

  --mode allocate  (ape-induction-budget)
      allocate(ctx) -> str : ONE shared budget covers BOTH proposal and dev eval.
      ctx tools:
        propose(n)     -> induce n candidates, charging n to the shared budget
        eval_on_dev(instr, rows) -> dev accuracy, charging each unique dev exec
        used()         -> current shared-budget spend
      Spending the whole budget proposing MANY candidates leaves ~0 dev eval each
      (blind/noisy pick); balancing a FEW proposals with ENOUGH dev eval each picks
      the candidate that generalizes. The harness ABORTS on budget overrun.

Emits canonical APE_FLOOR / APE_CHOSEN / APE_METRICS (test_acc + budget/used +
n_proposed).
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
    ap.add_argument("--mode", required=True, choices=["search", "allocate"])
    ap.add_argument("--dataset", required=True)      # agnews | sst2
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()
    ds = common.load_dataset(args.dataset)
    executor = common.Executor(ds, seed=args.seed)
    budget = int(args.budget)
    proposed = {"n": 0}

    import random as _r
    rng = _r.Random(args.seed)

    if args.mode == "search":
        search = common.load_surface(args.solution, "search")
        base_calls = executor.n_exec_calls

        def eval_on_dev(instr, rows):
            preds = executor.predict(instr, rows)
            if executor.n_exec_calls - base_calls > budget:
                raise BudgetExceeded(
                    f"dev budget {budget} exceeded")
            return common.accuracy(preds, rows)

        def induce(n):
            proposed["n"] += n
            return common.induce_instructions(executor, ds.pool, n, seed=args.seed)

        def paraphrase(s, n):
            return common.paraphrase_instruction(executor, s, n, seed=args.seed)

        ctx = {"executor": executor, "dataset": ds, "pool": ds.pool, "dev": ds.dev,
               "budget": budget, "eval_on_dev": eval_on_dev, "induce": induce,
               "paraphrase": paraphrase, "rng": rng}
        instruction = search(ctx)
        used = executor.n_exec_calls - base_calls
    else:
        allocate = common.load_surface(args.solution, "allocate")
        ctr = {"used": 0}

        def _charge(k):
            ctr["used"] += k
            if ctr["used"] > budget:
                raise BudgetExceeded(
                    f"shared budget {budget} exceeded (used {ctr['used']})")

        def propose(n):
            n = max(0, int(n))
            _charge(n)
            proposed["n"] += n
            return common.induce_instructions(executor, ds.pool, n, seed=args.seed)

        def eval_on_dev(instr, rows):
            before = executor.n_exec_calls
            preds = executor.predict(instr, rows)
            _charge(executor.n_exec_calls - before)
            return common.accuracy(preds, rows)

        ctx = {"executor": executor, "dataset": ds, "pool": ds.pool, "dev": ds.dev,
               "budget": budget, "propose": propose, "eval_on_dev": eval_on_dev,
               "used": lambda: ctr["used"], "rng": rng}
        instruction = allocate(ctx)
        used = ctr["used"]

    if not isinstance(instruction, str):
        raise SystemExit("surface must return an instruction STRING")

    dev_acc = executor.dev_accuracy(instruction, ds.dev)
    print(
        f"APE_SEARCH dev_accuracy={dev_acc:.9f} budget={budget} used={used} "
        f"n_proposed={proposed['n']}",
        flush=True,
    )
    common.emit_selected_metrics(
        executor,
        ds,
        instruction,
        t0,
        n_candidates=max(1, proposed["n"]),
        dev_exec_calls=executor.n_exec_calls,
    )


if __name__ == "__main__":
    main()
