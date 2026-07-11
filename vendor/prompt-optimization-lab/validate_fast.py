#!/usr/bin/env python3
"""Fast, STREAMING monotonicity validation for the ape-* tasks.

Same checks as validate.py but (a) loads the frozen LM ONCE per dataset and SHARES
the Executor across every baseline (its label-logprob cache + calibration cache make
repeated candidate/test evals free), and (b) flushes every line so progress is
visible under mlaunch's piped stdout. Prints, per dataset, the TEST accuracy of each
baseline so we can confirm the metric is MONOTONE in instruction quality:

  ape-instruction-search : empty  <  APE (LM-induced + hand-anchors, dev-selected)
  ape-candidate-scoring  : random <  exec-accuracy / log-likelihood estimator
  ape-search-strategy    : first / tiny-slice  <=  successive-halving  (<= oracle best)

Held-out discipline: selection uses only DEV; the reported accuracy is on the DISJOINT
TEST set, so a dev-overfit choice cannot win.
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


def P(*a):
    print(*a, flush=True)


def _load(path, attr):
    spec = importlib.util.spec_from_file_location("b", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, attr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="agnews,sst2")
    ap.add_argument("--budget", type=int, default=200)
    args = ap.parse_args()
    B = HERE / "baselines"
    R = __import__("random")

    for name in args.datasets.split(","):
        ds = common.load_dataset(name)
        floor = common.majority_prior_accuracy(ds.test)
        ex = common.Executor(ds)  # ONE executor, shared across all baselines below
        P(f"\n########## dataset={name}  majority_floor={floor:.4f} "
          f"pool={len(ds.pool)} dev={len(ds.dev)} test={len(ds.test)} ##########")

        # Pre-compute the fixed candidate pool + each candidate's TEST accuracy ONCE.
        cands = common.build_candidate_pool(ex, ds, 42)
        test_accs = [common.accuracy(ex.predict(c, ds.test), ds.test) for c in cands]
        best_test, worst_test = max(test_accs), min(test_accs)
        P(f"  [pool] n_cand={len(cands)} test_acc best={best_test:.4f} "
          f"worst={worst_test:.4f}")
        for c, a in sorted(zip(cands, test_accs), key=lambda t: -t[1]):
            P(f"     {a:.4f}  {c[:70]!r}")

        P("\n== ape-instruction-search ==")
        for tag, f in [("empty(weak)", "search_empty.py"),
                       ("ape(strong)", "search_ape.py")]:
            fn = _load(B / f, "optimize")
            ctx = {"executor": ex, "dataset": ds, "pool": ds.pool, "dev": ds.dev,
                   "rng": R.Random(42),
                   "induce_instructions": lambda n: common.induce_instructions(
                       ex, ds.pool, n, 42)}
            with redirect_stdout(io.StringIO()):
                instr = fn(ctx)
            dev = ex.dev_accuracy(instr, ds.dev)
            test = common.accuracy(ex.predict(instr, ds.test), ds.test)
            P(f"  {tag:14s} dev_acc={dev:.4f} TEST_acc={test:.4f}  "
              f"instr={(instr or '<EMPTY>')[:70]!r}")

        P("\n== ape-candidate-scoring ==")
        for tag, f in [("random(weak)", "scoring_random.py"),
                       ("execacc(strong)", "scoring_execacc.py"),
                       ("loglik(strong)", "scoring_loglik.py")]:
            fn = _load(B / f, "score_candidate")
            ctx = {"executor": ex, "dataset": ds, "dev": ds.dev, "pool": ds.pool,
                   "rng": R.Random(42)}
            scores = [float(fn(c, ctx)) for c in cands]
            bi = max(range(len(cands)), key=lambda i: scores[i])
            P(f"  {tag:16s} chosen_TEST={test_accs[bi]:.4f} "
              f"(pool best={best_test:.4f} worst={worst_test:.4f}) "
              f"instr={cands[bi][:60]!r}")

        P("\n== ape-search-strategy ==")
        for tag, f in [("first(weak)", "strategy_first.py"),
                       ("tiny(weak)", "strategy_tiny.py"),
                       ("halving(strong)", "strategy_halving.py")]:
            fn = _load(B / f, "select")
            # Clear the (instruction,input) prediction cache so the budget accounting
            # is REAL for this baseline (the harness uses a fresh executor per run;
            # here we share one executor, so we must reset its cache). The
            # calibration cache is FIXED infra and is intentionally kept.
            ex._cache = {}
            ex.n_exec_calls = 0
            budget = args.budget

            def eval_on_dev(instr, rows, _budget=budget):
                preds = ex.predict(instr, rows)
                if ex.n_exec_calls > _budget:
                    raise SystemExit(f"budget {_budget} exceeded (used {ex.n_exec_calls})")
                return common.accuracy(preds, rows)

            ctx = {"dataset": ds, "dev": ds.dev, "budget": budget,
                   "eval_on_dev": eval_on_dev, "rng": R.Random(42),
                   "n_candidates": len(cands)}
            with redirect_stdout(io.StringIO()):
                chosen = fn(cands, ctx)
            used = ex.n_exec_calls
            test = common.accuracy(ex.predict(chosen, ds.test), ds.test)
            P(f"  {tag:16s} chosen_TEST={test:.4f} (oracle best={best_test:.4f}) "
              f"used={used}/{budget} instr={chosen[:55]!r}")

    P("\nVALIDATE_DONE")


if __name__ == "__main__":
    main()
