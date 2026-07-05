#!/usr/bin/env python3
"""simp-no-repeat-ngram harness (fixed pipeline; ISOLATED from beam width).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, GREEDY decode (num_beams=1 FIXED, so the
lever is isolated from beam search), at the agent's NO-REPEAT NGRAM SIZE
(solution/norepeat.py -> build_no_repeat_ngram_size -> int). Scores corpus SARI per
setting.

no_repeat_ngram_size (Paulus et al. 2017 / Klein et al. 2017 OpenNMT-style) hard-
blocks any n-gram that has already appeared in the generated sequence (0 = off).
Isolated here under GREEDY decode so its effect is visible without beam search
masking it: a greedy T5 simplifier can otherwise loop on a repeated n-gram
indefinitely (wasting the length budget without adding new ADD/KEEP-credited
content), directly hurting SARI. A moderate block (n=3) removes loops without
over-constraining legitimate repeated function words; n=0 (off) is the weak default.

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

    build_n = common.load_surface(args.solution, "build_no_repeat_ngram_size")
    n = int(build_n())
    gen_kwargs = {
        "num_beams": 1,                 # FIXED (greedy — isolates from beam search)
        "repetition_penalty": 1.0,      # FIXED (isolates from repetition_penalty)
        "max_length": 128,              # FIXED
        "no_repeat_ngram_size": n,
    }
    print(f"SIMP_NOREPEAT no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']}", flush=True)

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
