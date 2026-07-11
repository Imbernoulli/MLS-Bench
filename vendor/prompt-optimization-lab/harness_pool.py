#!/usr/bin/env python3
"""ape pool-based harness (fixed pipeline) — candidate GENERATION and instruction
PARAPHRASE/REWRITE surfaces.

Frozen instruction LM, inference-only, zero-shot calibrated execution. The SELECTION
rule is FIXED here (highest DEV execution-accuracy), then the chosen instruction is
scored on the DISJOINT HELD-OUT TEST set. Only the PROPOSAL is editable:

  --surface propose  (ape-candidate-generation)
      propose(ctx) -> list[str] : generate the candidate instruction POOL. The harness
      dedupes it, dev-selects the best (fixed), test-evaluates. A thin/generic pool
      gives selection nothing good to pick; a diverse LM-induced pool from the labeled
      pool surfaces a genuinely good instruction.

  --surface rewrite  (ape-paraphrase-rewrite)
      rewrite(seed, ctx) -> list[str] : paraphrase a FIXED seed instruction. The pool
      scored is [seed] + your rewrites (deduped); dev-select best; test-evaluate. To
      beat the seed you must produce meaning-preserving rewrites the small LM follows
      better; returning the seed unchanged (or garbage) cannot improve over it.

Only the dev-selected instruction reaches the evaluation split. A terminal
``APE_RESULT`` line proves the complete inventory and single selected test pass.
"""
from __future__ import annotations

import argparse
import time

import common


# Fixed seed instructions the paraphrase surface rewrites (one per task). Chosen to
# be competent-but-plain so a good paraphrase has headroom and a degenerate rewrite
# (echo the seed) lands exactly at the seed's accuracy (the weak anchor).
_SEED = {
    "topic": "Classify the news article into its correct subject category.",
    # sentiment seed RECALIBRATED 2026-07-09 (probe2 ARM C, k1h20; numbers in
    # score_spec/leaderboard provenance): the previous seed ("Classify the review
    # as expressing a positive or negative opinion.") tested too strong under the
    # calibrated executor to serve as the weak anchor; this seed sits in the
    # intended weak band and the shipped strong rewrite retains clear headroom.
    "sentiment": "Determine the sentiment of this review.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=["propose", "rewrite"])
    ap.add_argument("--dataset", required=True)      # agnews | sst2
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()
    ds = common.load_dataset(args.dataset)
    executor = common.Executor(ds, seed=args.seed)
    fn = common.load_surface(args.solution, args.surface)

    import random as _r
    if args.surface == "propose":
        ctx = {
            "executor": executor, "dataset": ds, "pool": ds.pool, "dev": ds.dev,
            "rng": _r.Random(args.seed),
            "induce_instructions": lambda n: common.induce_instructions(
                executor, ds.pool, n, seed=args.seed),
        }
        candidates = [str(c).strip() for c in fn(ctx) if str(c).strip()]
    else:
        seed = _SEED[ds.task]
        ctx = {"executor": executor, "dataset": ds, "pool": ds.pool,
               "dev": ds.dev, "rng": _r.Random(args.seed), "seed": seed}
        rewrites = [str(c).strip() for c in fn(seed, ctx) if str(c).strip()]
        candidates = [seed] + rewrites

    # dedupe, preserve order; require at least one candidate
    seen, uniq = set(), []
    for c in candidates:
        k = c.lower()
        if k and k not in seen:
            seen.add(k); uniq.append(c)
    if not uniq:
        uniq = [""]

    chosen, _ = common.select_best_by_dev(executor, ds, uniq)
    common.emit_pool_metrics(executor, ds, uniq, chosen, t0)


if __name__ == "__main__":
    main()
