#!/usr/bin/env python3
"""simp-min-length harness (fixed pipeline; ISOLATED min-length floor).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using a FIXED beam width, length_penalty and
max_length (num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, max_length=96,
early_stopping=True all FIXED) and the agent's decoder-side min_length FLOOR
(solution/minlen.py -> build_min_length -> int). Scores corpus SARI per setting.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
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

    build_min_length = common.load_surface(args.solution, "build_min_length")
    ml = int(build_min_length())
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "length_penalty": 1.0,          # FIXED
        "max_length": 96,               # FIXED
        "early_stopping": True,         # FIXED
        "min_length": ml,
    }
    print(f"SIMP_MINLEN min_length={gen_kwargs['min_length']}", flush=True)

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
