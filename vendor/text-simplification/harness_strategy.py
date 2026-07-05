#!/usr/bin/env python3
"""simp-sampling-vs-beam harness (fixed pipeline; decoding STRATEGY choice).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's DECODING STRATEGY
(solution/strategy.py -> build_strategy -> a string):

  "sample" : plain multinomial sampling (do_sample=True, num_beams=1, temperature=1.0).
             No search at all — the weak floor.
  "topp"   : nucleus sampling (do_sample=True, num_beams=1, top_p=0.9). Restricts
             sampling to the model's likely tokens — better than plain sampling but
             still no search.
  "beam"   : deterministic beam search (num_beams=5, no_repeat_ngram_size=3). Proper
             search over the sequence probability — the strong setting.

All settings share the SAME fixed length window (max_length=128) and the SAME
frozen model; only the SEARCH STRATEGY varies. Proves that genuine search (beam)
beats sampling-based decoding for a metric that rewards precise ADD/KEEP/DELETE
edits (SARI), not just "any" plausible continuation.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = {"sample", "topp", "beam"}

_GEN = {
    "sample": {"do_sample": True, "num_beams": 1, "temperature": 1.0, "max_length": 128},
    "topp": {"do_sample": True, "num_beams": 1, "top_p": 0.9, "temperature": 1.0, "max_length": 128},
    "beam": {"num_beams": 5, "no_repeat_ngram_size": 3, "max_length": 128},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    build_strategy = common.load_surface(args.solution, "build_strategy")
    strategy = str(build_strategy()).strip()
    if strategy not in _VALID:
        raise SystemExit(f"strategy must be one of {sorted(_VALID)} (got {strategy!r})")
    gen_kwargs = dict(_GEN[strategy])
    print(f"SIMP_STRATEGY strategy={strategy}", flush=True)

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
    print(f"SIMP_DONE strategy={strategy} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
