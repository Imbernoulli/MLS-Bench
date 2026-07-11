#!/usr/bin/env python3
"""codegen-best-of-n-select harness (fixed pipeline).

A FROZEN small code LM draws a FIXED-size candidate pool per problem (n=8,
temperature 0.6, top_p 0.95 — all frozen here). The agent controls ONLY the
candidate SELECTION / reranking policy via
`select_candidate(candidates, problem, tok)`, which is given:
  * `candidates`: the list of extracted candidate programs (strings),
  * `problem`: dict incl. `visible_tests`, `entry_point`, `prompt` — the agent
    MAY execute candidates against the VISIBLE tests (via `common.run_tests` /
    `common.passes_all`) to filter/rerank,
  * `tok`: the tokenizer (for a logprob-style reranker if desired),
and returns the index of the chosen candidate.

The candidate pool, sampling, code extraction, and scoring are FIXED. Scoring is
on the DISJOINT HIDDEN tests. The agent MUST NOT be able to see the hidden
tests: the harness never passes them to the selector.

The verifier reports both selected pass@1 and the pool oracle rate. Candidate
signals are not assumed to have a particular ordering before full-protocol
anchors are measured.

Metric line:
    CG_METRICS task=codegen-best-of-n-select pass_at_1=<P> oracle_pass_at_1=<O> n=<N> elapsed=<T>
where pass_at_1 = (# problems whose SELECTED program passes ALL hidden tests)/n
and oracle_pass_at_1 = (# problems where ANY pool candidate passes all hidden
tests)/n  (the ceiling a perfect selector could reach; reported for diagnosis).
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "codegen-best-of-n-select"
# FIXED candidate-pool sampling (frozen — the agent does not control these).
POOL_N = 8
POOL_TEMPERATURE = 0.6
POOL_TOP_P = 0.95
POOL_MAX_NEW = 512


def _instruction(prob: dict) -> str:
    return (
        "Complete the following Python function. Return ONLY the complete "
        "function inside a single ```python code block.\n\n"
        f"{prob['prompt']}"
    )


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
    tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    select_candidate = common.load_surface(args.solution, "select_candidate")
    common.emit_protocol(TASK_ID, args.seed, problems)

    n = len(problems)
    n_hidden_ok = 0
    n_oracle = 0
    for i, prob in enumerate(problems):
        cfg = common.GenConfig(
            prompt=_instruction(prob),
            n_samples=POOL_N,
            do_sample=True,
            temperature=POOL_TEMPERATURE,
            top_p=POOL_TOP_P,
            max_new_tokens=POOL_MAX_NEW,
        )
        raw = common.generate(cfg, seed=args.seed)
        cands = [common.extract_code(r) for r in raw]

        # The agent's selector never receives the hidden tests.
        safe_prob = common.safe_problem(prob)
        idx = common.require_int(
            common.call_policy_without_generation(
                select_candidate, list(cands), safe_prob, tok
            ),
            "select_candidate()",
            0,
            len(cands) - 1,
        )
        chosen = cands[idx]

        hidden_ok = common.passes_all(
            chosen, prob["hidden_tests"], prob.get("test_setup", "")
        )
        oracle_ok = any(
            common.passes_all(c, prob["hidden_tests"], prob.get("test_setup", ""))
            for c in cands
        )
        n_hidden_ok += int(hidden_ok)
        n_oracle += int(oracle_ok)
        common.emit_item(i + 1, hidden_ok, oracle=oracle_ok)
        if i < 3:
            print(
                f"CG_SAMPLE i={i} id={prob['task_id']} idx={idx} "
                f"n_cands={len(cands)} hidden_ok={hidden_ok} oracle={oracle_ok}",
                flush=True,
            )
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden_ok)

    pass_at_1 = n_hidden_ok / n if n else 0.0
    oracle = n_oracle / n if n else 0.0
    dt = time.time() - t0
    print(
        f"CG_METRICS task={TASK_ID} pass_at_1={pass_at_1:.6f} "
        f"oracle_pass_at_1={oracle:.6f} "
        f"n={n} elapsed={dt:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
