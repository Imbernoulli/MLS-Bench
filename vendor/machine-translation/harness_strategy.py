#!/usr/bin/env python3
"""mt-decoding-strategy harness (fixed pipeline; the monotonicity / anti-gaming task).

For each complete pinned OPUS-100 source-to-English test split, produces a translation for every
source by the agent's DECODE STRATEGY (solution/strategy.py -> build_strategy ->
a string), then scores corpus sacreBLEU (and chrF) on the FIXED English
references.

The strategy chooses HOW the translation is produced:
  "beam"          : decode the FROZEN opus-mt-de-en with a fixed tuned config
                    (beam 5, length_penalty 1.0). The strong, real translation.
  "greedy"        : decode the FROZEN model greedily (beam 1). Real but weaker.
  "copy_source"   : DEGENERATE — return the German SOURCE unchanged. Wrong
                    language -> ~0 BLEU against the English references.
  "first_token"   : DEGENERATE — return only the first source word. ~0 BLEU.
  "empty"         : DEGENERATE — return an empty string. 0 BLEU.

This task exists to prove the metric is monotone and un-gameable: a copy-source /
constant / empty output must score clearly LOWER than the real model decode. Any
strategy is validated by corpus sacreBLEU against the FIXED references.

Emits one metric line:
    MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = {"beam", "greedy", "copy_source", "first_token", "empty"}

# Fixed tuned decode config for the "beam" strategy (agent cannot change it).
_BEAM_GEN = {
    "num_beams": 5,
    "length_penalty": 1.0,
    "max_new_tokens": 128,
    "early_stopping": True,
}
_GREEDY_GEN = {
    "num_beams": 1,
    "max_new_tokens": 128,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    strategy = common.load_surface_value(args.solution, "build_strategy")
    if not isinstance(strategy, str):
        raise TypeError("build_strategy must return a string")
    if strategy not in _VALID:
        raise SystemExit(f"strategy must be one of {sorted(_VALID)} (got {strategy!r})")
    print(f"MT_STRATEGY strategy={strategy}", flush=True)

    if strategy == "beam":
        model, tok = common.load_model_and_tokenizer(dev)
        preds = common.translate(model, tok, srcs, dict(_BEAM_GEN), dev)
    elif strategy == "greedy":
        model, tok = common.load_model_and_tokenizer(dev)
        preds = common.translate(model, tok, srcs, dict(_GREEDY_GEN), dev)
    elif strategy == "copy_source":
        preds = list(srcs)
    elif strategy == "first_token":
        preds = [(s.split()[0] if s.split() else "") for s in srcs]
    else:  # empty
        preds = ["" for _ in srcs]

    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
