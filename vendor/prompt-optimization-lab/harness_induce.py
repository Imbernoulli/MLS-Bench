#!/usr/bin/env python3
"""ape induction harness (fixed pipeline) — META-PROMPT design and FEW-SHOT
EXEMPLAR selection surfaces.

Frozen instruction LM, inference-only, zero-shot calibrated execution. Reverse-mode
induction (Honovich/Zhou) proposes candidate instructions by showing the frozen LM
labeled input/output examples; the harness then dev-selects (fixed highest dev
execution-accuracy) and test-evaluates on the DISJOINT HELD-OUT TEST set. Only the
induction CONDITIONING is editable:

  --surface select_exemplars  (ape-fewshot-exemplar)
      select_exemplars(pool, ctx) -> list[row] : choose WHICH labeled pool examples
      (and how many) condition induction. The harness induces one candidate from each
      of several deterministic shuffles of YOUR exemplar set, dev-selects, test-evals.
      A single/random exemplar yields a narrow/noisy induction; a small label-balanced
      diverse set yields a robust, generalizing instruction.

  --surface meta_prompt  (ape-meta-prompt)
      meta_prompt(examples, ctx) -> str : design the PROMPT (a format string with
      optional {demo} and {labels}) the frozen LM uses to INDUCE instructions. The
      harness shows several FIXED balanced example mini-sets and induces under YOUR
      meta-prompt, dev-selects, test-evals. A vague/empty meta-prompt elicits
      off-task text; a structured reverse-mode meta-prompt elicits clean task
      instructions.

Only the dev-selected instruction reaches the evaluation split. A terminal
``APE_RESULT`` line proves the complete inventory and single selected test pass.
"""
from __future__ import annotations

import argparse
import time

import common


def _balanced_minisets(ds, n_sets=4, k=5, seed=42):
    """FIXED balanced example mini-sets for meta-prompt induction (deterministic)."""
    import random
    rng = random.Random(seed)
    by = {}
    for r in ds.pool:
        by.setdefault(r["label"], []).append(r)
    labels = sorted(by)
    sets = []
    for s in range(n_sets):
        rows = []
        for i in range(k):
            lab = labels[(s + i) % len(labels)]
            rows.append(rng.choice(by[lab]))
        sets.append(rows)
    return sets


def _shuffles(rows, n=4, seed=42):
    import random
    out = []
    for s in range(n):
        r = random.Random(seed + s)
        c = list(rows); r.shuffle(c); out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True,
                    choices=["select_exemplars", "meta_prompt"])
    ap.add_argument("--dataset", required=True)      # agnews | sst2
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()
    ds = common.load_dataset(args.dataset)
    executor = common.Executor(ds, seed=args.seed)
    fn = common.load_surface(args.solution, args.surface)

    import random as _r
    candidates = []
    if args.surface == "select_exemplars":
        ctx = {"executor": executor, "dataset": ds, "dev": ds.dev,
               "rng": _r.Random(args.seed), "n_class": ds.n_class}
        exemplars = [r for r in fn(list(ds.pool), ctx)
                     if isinstance(r, dict) and "text" in r and "label" in r]
        if not exemplars:
            exemplars = list(ds.pool[:1])
        for sh in _shuffles(exemplars, 4, args.seed):
            candidates += common.induce_from_exemplars(
                executor, sh, 1, meta_prompt=None, seed=args.seed)
    else:
        # meta_prompt: give the surface one representative mini-set to condition its
        # template design; then induce under the returned template over FIXED sets.
        fixed_sets = _balanced_minisets(ds, 4, 5, args.seed)
        ctx = {"executor": executor, "dataset": ds, "pool": ds.pool,
               "dev": ds.dev, "rng": _r.Random(args.seed),
               "labels": ds.label_words()}
        tmpl = str(fn(fixed_sets[0], ctx))
        for rows in fixed_sets:
            candidates += common.induce_from_exemplars(
                executor, rows, 1, meta_prompt=tmpl, seed=args.seed)

    seen, uniq = set(), []
    for c in candidates:
        k = (c or "").strip().lower()
        if k and len(c.strip()) > 6 and k not in seen:
            seen.add(k); uniq.append(c.strip())
    if not uniq:
        uniq = [""]

    chosen, _ = common.select_best_by_dev(executor, ds, uniq)
    common.emit_pool_metrics(executor, ds, uniq, chosen, t0)


if __name__ == "__main__":
    main()
