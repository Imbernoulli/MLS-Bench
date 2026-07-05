#!/usr/bin/env python3
"""simp-length-penalty harness (fixed pipeline; ISOLATED length-penalty alpha).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using a FIXED beam width and length WINDOW
(num_beams=5, no_repeat_ngram_size=3, min_length=0, max_length=96, early_stopping=
True all FIXED) and the agent's beam-search length-PENALTY exponent
(solution/lenpenalty.py -> build_length_penalty -> float). Scores corpus SARI per
setting.

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

    build_lp = common.load_surface(args.solution, "build_length_penalty")
    lp = float(build_lp())
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "min_length": 0,                # FIXED
        "max_length": 96,               # FIXED
        "early_stopping": True,         # FIXED
        "length_penalty": lp,
    }
    print(f"SIMP_LENPENALTY length_penalty={gen_kwargs['length_penalty']}", flush=True)

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
