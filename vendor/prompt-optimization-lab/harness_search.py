#!/usr/bin/env python3
"""ape-instruction-search harness (fixed pipeline).

Frozen instruction LM, inference-only, ZERO-SHOT execution (no demonstrations in
the executed prompt). The agent-editable surface `optimize(ctx)` returns a FINAL
instruction string; it may PROPOSE candidates (fixed generic string vs LM-induced
from the labeled pool vs iterative resampling) and SELECT among them using the DEV
set. The harness then scores the chosen instruction on the DISJOINT HELD-OUT TEST
set by the FIXED forced-choice executor. The base LM, the executor, the proposal
pool, the dev/test split, and the label set are all FIXED here.

Emits:
    APE_FLOOR majority_prior_acc=<F> dataset=<D>
    APE_CHOSEN instruction="<...>"
    APE_METRICS test_acc=<A> dev_acc=<A2> dataset=<D> n_test=<N> exec_calls=<C> elapsed=<T>
"""
from __future__ import annotations

import argparse
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
    optimize = common.load_surface(args.solution, "optimize")

    import random as _r
    ctx = {
        "executor": executor,
        "dataset": ds,
        "pool": ds.pool,          # labeled few-shot examples for PROPOSAL
        "dev": ds.dev,            # small set for SCORING/SELECTING candidates
        "rng": _r.Random(args.seed),
        "induce_instructions": lambda n: common.induce_instructions(
            executor, ds.pool, n, seed=args.seed),
    }

    instruction = optimize(ctx)
    if not isinstance(instruction, str):
        raise SystemExit("optimize(ctx) must return a single instruction STRING")

    # Report the dev accuracy of the chosen instruction (diagnostic) and the
    # HELD-OUT TEST accuracy (the scored metric). Test executions are NOT part of
    # the agent's search — they measure generalization of the chosen instruction.
    dev_acc = executor.dev_accuracy(instruction, ds.dev)
    search_calls = executor.n_exec_calls
    print(f"APE_DEV selected_accuracy={dev_acc:.9f}", flush=True)
    common.emit_selected_metrics(
        executor,
        ds,
        instruction,
        t0,
        n_candidates=1,
        dev_exec_calls=search_calls,
    )


if __name__ == "__main__":
    main()
