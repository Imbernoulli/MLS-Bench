#!/usr/bin/env python3
"""cd-numeric-answer harness (fixed pipeline).

A FROZEN small instruction LM answers the FULL pinned GSM8K test split.
The agent controls ONLY the decoding policy via `build_decoder(question, tok)`
which returns a `common.DecodeSpec` describing:
  * the prompt (instruction / schema wording),
  * WHAT to constrain (the whole answer vs an answer field after free reasoning),
  * the answer constraint (a regex the numeric answer must match), and
  * whether a free-form reasoning preamble is allowed before the answer is
    constrained (reason-first vs answer-only).

The model, gold answers, greedy decoding, token budget, and the FIXED numeric
extractor / correctness check are all frozen here.

A sample counts as CORRECT only if the answer region is structurally VALID
*and* the extracted integer equals the gold integer — so a decoder that emits a
valid-but-wrong answer gains nothing over one that emits garbage.

Emits one metric line:
    CD_METRICS protocol=<P> task=<T> surface=<S> dataset=gsm8k
               valid_rate=<V> accuracy=<A> n=<N> elapsed=<T>
where accuracy = (# valid AND correct) / n.
"""
from __future__ import annotations

import argparse
import re
import time

import common


PROTOCOL = "constrained-decoding-full-v3"
TASK_SURFACES = {
    "cd-numeric-answer": "decoder_numeric",
    "cd-numeric-budget": "decoder_budget",
    "cd-numeric-format": "decoder_format",
    "cd-numeric-json": "decoder_json",
    "cd-numeric-prefix": "decoder_prefix",
    "cd-numeric-repair": "decoder_repair",
    "cd-numeric-trigger": "decoder_trigger",
}


def _validate_identity(task_id: str, surface: str) -> None:
    if re.fullmatch(r"cd-[a-z0-9-]+", task_id) is None:
        raise ValueError("invalid constrained-decoding task identity")
    if re.fullmatch(r"decoder_[a-z0-9_]+", surface) is None:
        raise ValueError("invalid constrained-decoding surface identity")
    if TASK_SURFACES.get(task_id) != surface:
        raise ValueError(
            f"task/surface mismatch: task={task_id!r} surface={surface!r}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1319)
    args = ap.parse_args()

    _validate_identity(args.task_id, args.surface)

    common.set_seeds(args.seed)
    t0 = time.time()

    tok, model = common.load_model()
    print(
        f"CD_MODEL protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} "
        f"params={sum(parameter.numel() for parameter in model.parameters())} "
        f"device={next(model.parameters()).device} dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    build_decoder = common.load_surface(args.solution, "build_decoder")
    items = common.load_gsm8k(args.n)
    print(
        f"CD_DATA protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} dataset=gsm8k n={len(items)} seed={args.seed}",
        flush=True,
    )

    n = len(items)
    n_valid = 0
    n_correct = 0
    for i, it in enumerate(items):
        spec = build_decoder(it["question"], tok)
        res = common.run_decode(spec, seed=args.seed)
        valid = res["valid"]
        pred = common.extract_gsm8k_answer(res["answer_text"]) if valid else None
        gold = common.normalize_gold_int(it["gold"])
        correct = valid and pred is not None and pred == gold
        n_valid += int(valid)
        n_correct += int(correct)
        if i < 3:
            print(
                f"CD_SAMPLE i={i} valid={valid} pred={pred} gold={gold} "
                f"correct={correct} answer={res['answer_text'][:40]!r}",
                flush=True,
            )
        if (i + 1) % 50 == 0:
            print(
                f"CD_PROGRESS {i+1}/{n} valid={n_valid} correct={n_correct}",
                flush=True,
            )

    valid_rate = n_valid / n if n else 0.0
    accuracy = n_correct / n if n else 0.0
    dt = time.time() - t0
    print(
        f"CD_METRICS protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} dataset=gsm8k "
        f"valid_rate={valid_rate:.6f} accuracy={accuracy:.6f} "
        f"n={n} elapsed={dt:.1f}",
        flush=True,
    )
    print(
        f"CD_COMPLETE protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} dataset=gsm8k n={n} seed={args.seed} status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
