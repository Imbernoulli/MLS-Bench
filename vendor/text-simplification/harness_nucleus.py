#!/usr/bin/env python3
"""simp-nucleus-topp harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using SAMPLING (do_sample=True, num_beams=1,
temperature=1.0 all FIXED) restricted to the agent's NUCLEUS (top-p)
(solution/nucleus.py -> build_top_p -> float). Scores corpus SARI per setting.

Nucleus (top-p) sampling truncates the sampling distribution to the smallest set of
tokens whose cumulative probability >= p, then samples from that renormalized set.
A WIDE nucleus (p close to 1.0) samples from (almost) the full vocabulary -> noisy,
off-distribution tokens -> lower SARI; a TIGHT nucleus (p small) restricts sampling
to the model's most probable tokens -> closer to the model's mode -> higher SARI.
Isolated from temperature/beam (both FIXED) so only top-p is visible.

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

    build_top_p = common.load_surface(args.solution, "build_top_p")
    top_p = float(build_top_p())
    gen_kwargs = {
        "do_sample": True,           # FIXED (sampling, not beam search)
        "num_beams": 1,              # FIXED
        "temperature": 1.0,          # FIXED
        "no_repeat_ngram_size": 3,   # FIXED
        "max_length": 128,           # FIXED
        "top_p": top_p,
    }
    print(f"SIMP_NUCLEUS top_p={gen_kwargs['top_p']}", flush=True)

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
