#!/usr/bin/env python3
"""End-to-end monotonicity validation for the prompt-optimization-lab (ape-*) tasks.

Runs each harness with its WEAK and STRONG baseline(s) on both datasets and prints
the TEST accuracies so we can confirm the metric is MONOTONE in instruction quality:

  ape-instruction-search : empty  <  APE (LM-induced + dev-selected)
  ape-candidate-scoring  : random <  exec-accuracy / log-likelihood estimator
  ape-search-strategy    : first / tiny-slice  <  successive-halving  (<= oracle best)

Held-out discipline: selection uses only DEV; the reported accuracy is on the
DISJOINT TEST set, so a dev-overfit choice cannot win.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def _load(path, attr):
    spec = importlib.util.spec_from_file_location("b", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, attr)


def run_search(ds, fn):
    ex = common.Executor(ds)
    ctx = {"executor": ex, "dataset": ds, "pool": ds.pool, "dev": ds.dev,
           "rng": __import__("random").Random(42),
           "induce_instructions": lambda n: common.induce_instructions(ex, ds.pool, n, 42)}
    instr = fn(ctx)
    dev = ex.dev_accuracy(instr, ds.dev)
    test = common.accuracy(ex.predict(instr, ds.test), ds.test)
    return dev, test, (instr or "<EMPTY>")[:80]


def run_scoring(ds, fn):
    ex = common.Executor(ds)
    cands = common.build_candidate_pool(ex, ds, 42)
    ctx = {"executor": ex, "dataset": ds, "dev": ds.dev, "pool": ds.pool,
           "rng": __import__("random").Random(42)}
    scores = [float(fn(c, ctx)) for c in cands]
    best_i = max(range(len(cands)), key=lambda i: scores[i])
    test_accs = [common.accuracy(ex.predict(c, ds.test), ds.test) for c in cands]
    return test_accs[best_i], max(test_accs), min(test_accs), cands[best_i][:80]


def run_strategy(ds, fn, budget=200):
    ex = common.Executor(ds)
    cands = common.build_candidate_pool(ex, ds, 42)
    ex.n_exec_calls = 0

    def eval_on_dev(instr, rows):
        preds = ex.predict(instr, rows)
        if ex.n_exec_calls > budget:
            raise SystemExit(f"budget {budget} exceeded (used {ex.n_exec_calls})")
        return common.accuracy(preds, rows)

    ctx = {"dataset": ds, "dev": ds.dev, "budget": budget,
           "eval_on_dev": eval_on_dev, "rng": __import__("random").Random(42),
           "n_candidates": len(cands)}
    chosen = fn(cands, ctx)
    used = ex.n_exec_calls
    test = common.accuracy(ex.predict(chosen, ds.test), ds.test)
    best = max(common.accuracy(ex.predict(c, ds.test), ds.test) for c in cands)
    return test, best, used, chosen[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="agnews,sst2")
    ap.add_argument("--budget", type=int, default=200)
    args = ap.parse_args()
    B = HERE / "baselines"

    for name in args.datasets.split(","):
        ds = common.load_dataset(name)
        floor = common.majority_prior_accuracy(ds.test)
        print(f"\n########## dataset={name}  majority_floor={floor:.4f} "
              f"pool={len(ds.pool)} dev={len(ds.dev)} test={len(ds.test)} ##########")

        print("\n== ape-instruction-search ==")
        for tag, f in [("empty(weak)", "search_empty.py"),
                       ("ape(strong)", "search_ape.py")]:
            fn = _load(B / f, "optimize")
            buf = io.StringIO()
            with redirect_stdout(buf):
                dev, test, ins = run_search(ds, fn)
            print(f"  {tag:14s} dev_acc={dev:.4f} TEST_acc={test:.4f}  instr={ins!r}")

        print("\n== ape-candidate-scoring ==")
        for tag, f in [("random(weak)", "scoring_random.py"),
                       ("execacc(strong)", "scoring_execacc.py"),
                       ("loglik(strong)", "scoring_loglik.py")]:
            fn = _load(B / f, "score_candidate")
            buf = io.StringIO()
            with redirect_stdout(buf):
                test, best, worst, ins = run_scoring(ds, fn)
            print(f"  {tag:16s} chosen_TEST={test:.4f} (pool best={best:.4f} "
                  f"worst={worst:.4f}) instr={ins!r}")

        print("\n== ape-search-strategy ==")
        for tag, f in [("first(weak)", "strategy_first.py"),
                       ("tiny(weak)", "strategy_tiny.py"),
                       ("halving(strong)", "strategy_halving.py")]:
            fn = _load(B / f, "select")
            buf = io.StringIO()
            with redirect_stdout(buf):
                test, best, used, ins = run_strategy(ds, fn, args.budget)
            print(f"  {tag:16s} chosen_TEST={test:.4f} (oracle best={best:.4f}) "
                  f"used={used}/{args.budget} instr={ins!r}")

    print("\nVALIDATE_DONE")


if __name__ == "__main__":
    main()
