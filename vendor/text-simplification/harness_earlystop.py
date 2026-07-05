#!/usr/bin/env python3
"""simp-early-stopping harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier under FIXED beam search (num_beams=5,
no_repeat_ngram_size=3, length_penalty=1.0), varying ONLY the agent's EARLY
STOPPING policy (solution/earlystop.py -> build_early_stopping ->
True / False / "never"). Scores corpus SARI per setting.

transformers `model.generate(early_stopping=...)` controls when beam search halts:
  False    : keep searching until max_length is always reached (or all beams end)
             — heuristic OFF, matches the older HF default; can run needlessly long
             and let low-quality continuations creep in.
  True     : stop as soon as `num_beams` finished (EOS-terminated) hypotheses exist
             — the standard efficient/quality heuristic.
  "never"  : stop only when it is provably impossible to find a better hypothesis
             (canonical beam-search stopping condition) — the exhaustive variant.
Isolated from the length window (max_length FIXED at 128) and the beam/repetition
config (FIXED) so only the STOPPING POLICY varies.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = {False, True, "never"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    build_es = common.load_surface(args.solution, "build_early_stopping")
    es = build_es()
    if es not in _VALID:
        raise SystemExit(f"early_stopping must be one of {sorted(map(str, _VALID))} (got {es!r})")
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "length_penalty": 1.0,          # FIXED
        "max_length": 128,              # FIXED
        "early_stopping": es,
    }
    print(f"SIMP_EARLYSTOP early_stopping={es}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = common.simplify(model, tok, srcs, dict(gen_kwargs), dev)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        print(f"SIMP_METRICS setting={setting} sari={sari:.6f} bleu={bleu:.4f} "
              f"n_sents={len(srcs)} plen={plen:.1f} lenratio={lr:.3f}", flush=True)

    dt = time.time() - t0
    print(f"SIMP_DONE elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
