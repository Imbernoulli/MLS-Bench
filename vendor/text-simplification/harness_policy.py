#!/usr/bin/env python3
"""simp-source-policy harness (fixed pipeline; the monotonicity / anti-gaming task).

For EACH of the THREE FIXED simplification test settings (asset / turk / wiki),
produces a rewrite for every source sentence by the agent's REWRITE POLICY
(solution/policy.py -> build_policy -> a string), then scores corpus SARI on that
setting's FIXED multi-reference set.

The policy chooses HOW the rewrite is produced:
  "beam"        : simplify the FROZEN t5-base with a fixed tuned config (beam 5,
                  no-repeat-3gram). The strong, real simplification (SOTA-style).
  "greedy"      : simplify the FROZEN model greedily (beam 1). Real but weaker.
  "truncate"    : TRUNCATION baseline — keep the first 75% of the words (delete the
                  tail). A naive deletion heuristic (reported diagnostic).
  "first_token" : DEGENERATE FLOOR — return only the first source word. Low SARI.
  "empty"       : DEGENERATE FLOOR — return an empty string. Low SARI.

Proves the metric is monotone and un-gameable across ALL THREE settings: a
meaning-destroying output scores a genuinely LOW SARI on every setting, and only a
real T5 simplifier reaches the SOTA-scale top:
    empty/first_token (~11-20)  <  greedy (~33-35)  <  beam (~43-45 = SOTA scale)

NOTE on `copy_input`: the pure identity baseline scores HIGH SARI (~52-60) on the
conservative ASSET/Turk references — a well-documented SARI-KEEP artifact (the human
references keep most source n-grams, so copying earns large KEEP credit). SARI's
ADD/DELETE terms, not KEEP, are what a real simplifier must win. To keep this
monotonicity task un-gameable, `copy_input` is NOT a selectable policy here (the
anchor sits between the meaning-destroying FLOOR and the real model); the length /
DELETE-balance lever is exercised by the sibling simp-length-control task instead.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = {"beam", "greedy", "truncate", "first_token", "empty"}

# Fixed tuned decode config for the "beam" policy (agent cannot change it).
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
    t0 = time.time()

    build_policy = common.load_surface(args.solution, "build_policy")
    policy = str(build_policy()).strip()
    if policy not in _VALID:
        raise SystemExit(f"policy must be one of {sorted(_VALID)} (got {policy!r})")
    print(f"SIMP_POLICY policy={policy}", flush=True)

    model_tok = None
    if policy in ("beam", "greedy"):
        model_tok = common.load_model_and_tokenizer(dev)

    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = _predict(policy, srcs, dev, model_tok)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        print(f"SIMP_METRICS setting={setting} sari={sari:.6f} bleu={bleu:.4f} "
              f"n_sents={len(srcs)} plen={plen:.1f} lenratio={lr:.3f}", flush=True)

    dt = time.time() - t0
    print(f"SIMP_DONE policy={policy} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
