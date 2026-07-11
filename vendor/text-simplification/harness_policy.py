#!/usr/bin/env python3
"""simp-source-policy harness (fixed pipeline).

For EACH of the THREE FIXED simplification test settings (asset / turk / wiki),
produces a rewrite for every source sentence by the agent's REWRITE POLICY
(solution/policy.py -> build_policy -> a string), then scores corpus SARI on that
setting's FIXED multi-reference set.

The policy chooses HOW the rewrite is produced:
  "beam"        : simplify the FROZEN t5-base with a fixed multi-beam config and
                  no-repeat n-gram blocking.
  "greedy"      : simplify the FROZEN model greedily.
  "truncate"    : keep the first 75% of the words.
  "first_token" : return only the first source word.
  "empty"       : return an empty string.

The pure identity policy is not selectable in this task; only the five
strings accepted by build_policy() are evaluated.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-source-policy"
SURFACE = "policy"

_VALID = {"beam", "greedy", "truncate", "first_token", "empty"}

# Fixed decode config for the "beam" policy (agent cannot change it).
_BEAM_GEN = {
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "max_length": 128,
    "length_penalty": 1.0,
    "early_stopping": True,
}
_GREEDY_GEN = {"num_beams": 1, "max_length": 128}


def _predict(policy, srcs, dev, model_tok):
    if policy == "beam":
        model, tok = model_tok
        return common.simplify(model, tok, srcs, dict(_BEAM_GEN), dev)
    if policy == "greedy":
        model, tok = model_tok
        return common.simplify(model, tok, srcs, dict(_GREEDY_GEN), dev)
    if policy == "truncate":
        return common.truncate_tail(srcs, keep_ratio=0.75)
    if policy == "first_token":
        return [(s.split()[0] if s.split() else "") for s in srcs]
    return ["" for _ in srcs]  # empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_policy = common.load_surface(args.solution, "build_policy")
    policy = common.require_surface_choice(
        build_policy(), "policy", _VALID, surface="build_policy"
    )
    print(f"SIMP_POLICY policy={policy}", flush=True)

    model_tok = None
    if policy in ("beam", "greedy"):
        model_tok = common.load_model_and_tokenizer(dev)

    metric_lines = []
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = _predict(policy, srcs, dev, model_tok)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        metric_lines.append(common.emit_metrics(
            task=TASK_ID, surface=SURFACE, setting=setting, sari=sari,
            bleu=bleu, n_sents=len(srcs), plen=plen, lenratio=lr,
        ))

    common.emit_done(
        task=TASK_ID, surface=SURFACE, seed=args.seed,
        model_choice="base_turk", metric_lines=metric_lines,
        elapsed=time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
