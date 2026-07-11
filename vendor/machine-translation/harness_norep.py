#!/usr/bin/env python3
"""mt-no-repeat-ngram harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model using a FIXED
beam width (5) and a FIXED length policy; the agent controls ONLY the
`no_repeat_ngram_size` repetition block (solution/norep.py -> build_norep_config
-> {no_repeat_ngram_size}). Scores corpus sacreBLEU / chrF.

Blocking repeated n-grams (Paulus et al. 2017; Klein et al. 2017 OpenNMT) stops
degenerate loops; too small a value (1-2) forbids legitimate repeated words and
hurts, too large (>=5) never triggers. The standard MT value is 3.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    cfg = common.require_config(
        common.load_surface_value(args.solution, "build_norep_config"),
        "build_norep_config",
        {"no_repeat_ngram_size"},
    )
    nrs = common.require_int(
        cfg["no_repeat_ngram_size"], "no_repeat_ngram_size", 0, 10
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "length_penalty": 1.0,          # FIXED
        "early_stopping": True,         # FIXED
        "max_new_tokens": 128,          # FIXED
        "no_repeat_ngram_size": nrs,
    }
    print(f"MT_NOREP no_repeat_ngram_size={nrs}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
