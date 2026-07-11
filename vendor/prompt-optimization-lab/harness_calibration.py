#!/usr/bin/env python3
"""ape calibration harness (fixed pipeline) — CALIBRATION-INPUT selection surface
(Calibrate-Before-Use, Zhao et al. 2021).

Frozen instruction LM, inference-only, zero-shot calibrated execution. The candidate
POOL is FIXED (index 0 is a distractor). The HELD-OUT TEST metric is ALWAYS computed
with the FIXED content-free "N/A" calibration (Executor.predict) — that never moves.
What IS editable is the calibration used to form the DEV SELECTION signal:

  calibration_inputs(ctx) -> list[str] : the content-free input(s) whose mean label
  distribution is subtracted when scoring each candidate on DEV to pick the best.

A poor choice (e.g. a real task sentence, or the label words themselves) does NOT
debias — it injects a label prior that mis-ranks candidates and selects a distractor,
so TEST (fixed "N/A") accuracy stays near the floor. A good set of genuinely
content-free inputs (e.g. "N/A", "", "the", "a") yields a stable dev ranking that
surfaces the candidate which also wins on the held-out test.

Only the dev-selected instruction reaches the evaluation split. The terminal
``APE_RESULT`` uses the fixed "N/A" calibration and proves the complete inventory
and single selected test pass; the editable calibration only affects selection.
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
    calibration_inputs = common.load_surface(args.solution, "calibration_inputs")

    candidates = common.build_candidate_pool(executor, ds, seed=args.seed)
    executor.n_exec_calls = 0  # pool build is fixed infrastructure

    import random as _r
    ctx = {"executor": executor, "dataset": ds, "dev": ds.dev, "pool": ds.pool,
           "rng": _r.Random(args.seed), "n_candidates": len(candidates)}
    cal_inputs = [str(x) for x in calibration_inputs(ctx)]

    # rank candidates by AGENT-calibrated dev accuracy; pick argmax
    best_i, best_s = 0, float("-inf")
    for i, c in enumerate(candidates):
        s = common.calibrated_dev_accuracy(executor, c, ds.dev, cal_inputs)
        if s > best_s:
            best_s, best_i = s, i
    chosen = candidates[best_i]

    cal_show = "|".join(cal_inputs)[:60]
    print(f'APE_CALIBRATION inputs="{cal_show}"', flush=True)
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
