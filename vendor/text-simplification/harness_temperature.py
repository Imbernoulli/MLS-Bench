#!/usr/bin/env python3
"""simp-decoding-temperature harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using SAMPLING (do_sample=True, num_beams=1
FIXED) at the agent's TEMPERATURE (solution/temperature.py -> build_temperature ->
float). Scores corpus SARI per setting.

Temperature reshapes the softmax before sampling: HIGH temperature (>1) flattens
the distribution towards uniform -> more random, less faithful tokens -> lower
SARI; LOW temperature (<1) sharpens it towards the model's mode (closer to greedy)
-> higher SARI. Isolated from beam search (num_beams FIXED at 1, sampling only) so
only the temperature lever is visible.

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

    build_temperature = common.load_surface(args.solution, "build_temperature")
    temperature = float(build_temperature())
    gen_kwargs = {
        "do_sample": True,           # FIXED (sampling, not beam search)
        "num_beams": 1,              # FIXED
        "no_repeat_ngram_size": 3,   # FIXED (avoid degenerate loops)
        "max_length": 128,           # FIXED
        "temperature": temperature,
    }
    print(f"SIMP_TEMPERATURE temperature={gen_kwargs['temperature']}", flush=True)

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
