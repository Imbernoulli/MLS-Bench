#!/usr/bin/env python3
"""codegen-sampling-strategy harness (fixed pipeline).

A frozen code LM writes a Python function for each problem. The editable
`sampling_parameters(problem)` returns only `(temperature, top_p)`. The prompt,
eight-candidate pool, 512-token cap, model, extraction, and selection are fixed.

  * candidate SELECTION is fixed: among the drawn candidates, keep those that
    pass ALL VISIBLE example tests (execute-and-filter on the visible tests);
    pick the first survivor; if none survive, fall back to the first candidate.
  * code EXTRACTION is fixed (first ```python fenced block, else whole text).
  * scoring is on the DISJOINT HIDDEN tests.

No temperature or nucleus cutoff is assumed to be best before full-protocol
anchors are measured.

Metric line:
    CG_METRICS task=codegen-sampling-strategy pass_at_1=<P> visible_solve_rate=<V> n=<N> elapsed=<T>
where pass_at_1 = (# problems whose FINAL chosen program passes ALL hidden
tests) / n.
"""
from __future__ import annotations

import argparse
import time

import common


TASK_ID = "codegen-sampling-strategy"
POOL_N = 8
MAX_NEW_TOKENS = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=common.EXPECTED_PROBLEMS)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()

    # Fork the executor server BEFORE CUDA is initialized (see common.py).
    common.start_executor()
    _tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    sampling_parameters = common.load_surface(args.solution, "sampling_parameters")
    common.emit_protocol(TASK_ID, args.seed, problems)

    n = len(problems)
    n_hidden_ok = 0
    n_visible_ok = 0
    for i, prob in enumerate(problems):
        policy_problem = {
            "task_id": prob["task_id"],
            "entry_point": prob["entry_point"],
            "prompt": prob["prompt"],
        }
        params = common.call_policy_without_generation(
            sampling_parameters, policy_problem
        )
        if type(params) is not tuple or len(params) != 2:
            raise TypeError("sampling_parameters() must return (temperature, top_p)")
        temperature = common.require_real(params[0], "temperature")
        top_p = common.require_real(params[1], "top_p")
        if not 0.0 < temperature <= 2.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("sampling parameters are outside the fixed envelope")
        cfg = common.GenConfig(
            prompt=(
                "Complete the following Python function. Return ONLY the complete "
                "function inside a single ```python code block.\n\n" + prob["prompt"]
            ),
            n_samples=POOL_N,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        raw_cands = common.generate(cfg, seed=args.seed)
        cands = [common.extract_code(r) for r in raw_cands]

        # FIXED selection: execute-and-filter on the VISIBLE tests, first survivor.
        chosen = cands[0]
        for c in cands:
            if common.passes_all(c, prob["visible_tests"], prob.get("test_setup", "")):
                chosen = c
                break

        visible_ok = common.passes_all(
            chosen, prob["visible_tests"], prob.get("test_setup", "")
        )
        hidden_ok = common.passes_all(
            chosen, prob["hidden_tests"], prob.get("test_setup", "")
        )
        n_visible_ok += int(visible_ok)
        n_hidden_ok += int(hidden_ok)
        common.emit_item(i + 1, hidden_ok, visible=visible_ok)
        if i < 3:
            print(
                f"CG_SAMPLE i={i} id={prob['task_id']} n_cands={len(cands)} "
                f"visible_ok={visible_ok} hidden_ok={hidden_ok}",
                flush=True,
            )
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden_ok)

    pass_at_1 = n_hidden_ok / n if n else 0.0
    visible_rate = n_visible_ok / n if n else 0.0
    dt = time.time() - t0
    print(
        f"CG_METRICS task={TASK_ID} pass_at_1={pass_at_1:.6f} "
        f"visible_solve_rate={visible_rate:.6f} "
        f"n={n} elapsed={dt:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
