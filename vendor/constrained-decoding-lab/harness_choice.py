#!/usr/bin/env python3
"""cd-forced-choice harness (fixed pipeline).

A FROZEN small instruction LM classifies the full official AG News test split
of short texts into a
FIXED label set. The agent controls ONLY the decoding policy via
`build_decoder(text, labels, tok)` returning a `common.DecodeSpec`:
  * the prompt,
  * WHAT to constrain — either constrain the output directly to the label set
    (`choices=labels`, guaranteed valid) or let the model free-generate and then
    map/constrain, and
  * whether a reasoning preamble precedes the constrained label.

The model, gold labels, decoding, and correctness check are frozen here.

A sample is CORRECT only if the committed label is a member of the label set
(VALID) *and* equals the gold label. A degenerate decoder that always returns
one constant valid label therefore scores at the majority-class rate (low),
never the correctness of a real classifier.

Emits:
    CD_METRICS protocol=<P> task=<T> surface=<S> dataset=agnews
               valid_rate=<V> accuracy=<A> n=<N> elapsed=<T>
"""
from __future__ import annotations

import argparse
import re
import time

import common


PROTOCOL = "constrained-decoding-full-v3"
TASK_SURFACES = {
    "cd-choice-reasoning": "decoder_choice_reasoning",
    "cd-choice-verbalizer": "decoder_choice_verbalizer",
    "cd-forced-choice": "decoder_choice",
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
    ap.add_argument("--n", type=int, default=7600)
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
    items, labels = common.load_classification(args.n)
    print(
        f"CD_DATA protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} dataset=agnews n={len(items)} seed={args.seed}",
        flush=True,
    )
    label_set = set(labels)

    n = len(items)
    n_valid = 0
    n_correct = 0
    for i, it in enumerate(items):
        spec = build_decoder(it["text"], list(labels), tok)
        res = common.run_decode(spec, seed=args.seed)
        committed = res["answer_text"].strip()
        if spec.choice_labels is not None:
            if (len(spec.choice_labels) != len(labels)
                    or len(set(spec.choice_labels)) != len(labels)
                    or set(spec.choice_labels) != label_set):
                raise ValueError(
                    "choice_labels must map one-to-one onto the complete fixed label set"
                )
            pred = res.get("mapped_answer")
            if pred not in label_set:
                raise ValueError("decoder returned an invalid mapped choice label")
        else:
            # Without an explicit verbalizer map, only a canonical label is valid.
            pred = _match_label(committed, labels)
        valid = pred is not None
        gold = str(it["gold"]).strip()
        correct = valid and pred == gold
        n_valid += int(valid)
        n_correct += int(correct)
        if i < 3:
            print(
                f"CD_SAMPLE i={i} valid={valid} pred={pred!r} gold={gold!r} "
                f"correct={correct} committed={committed[:40]!r}",
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
        f"surface={args.surface} dataset=agnews "
        f"valid_rate={valid_rate:.6f} accuracy={accuracy:.6f} "
        f"n={n} elapsed={dt:.1f}",
        flush=True,
    )
    print(
        f"CD_COMPLETE protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} dataset=agnews n={n} seed={args.seed} status=ok",
        flush=True,
    )


def _match_label(committed: str, labels: list[str]) -> str | None:
    """Return the exact label the committed text equals (case-insensitive), or
    None. Requires an exact, whole-string label match — extra prose around it
    does NOT count as valid, which is the whole point of a forced-choice
    constraint."""
    c = committed.strip().strip(".").strip().lower()
    for lab in labels:
        if c == lab.lower():
            return lab
    return None


if __name__ == "__main__":
    main()
