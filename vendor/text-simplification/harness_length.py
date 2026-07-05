#!/usr/bin/env python3
"""simp-length-control harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's LENGTH / COMPRESSION decode
config (solution/length.py -> build_length_config -> {min_length, max_length,
length_penalty}). num_beams and no_repeat_ngram_size are FIXED, so ONLY the length
window varies. Scores corpus SARI per setting.

Length is a direct lever on the DELETE/ADD balance SARI measures: simplification
usually SHORTENS a sentence, so a runaway-long decode (large length_penalty / large
max_length) keeps everything (acts like copy-the-input -> few DELETE credits ->
lower SARI), while a sensibly compressive window recovers the edits. Over-compress
and meaning (ADD/KEEP) collapses.

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

    build_length_config = common.load_surface(args.solution, "build_length_config")
    cfg = build_length_config()
    gen_kwargs = {
        "num_beams": 5,                 # FIXED (strong beam)
        "no_repeat_ngram_size": 3,      # FIXED
        "early_stopping": True,         # FIXED
        "min_length": int(cfg.get("min_length", 0)),
        "max_length": int(cfg.get("max_length", 128)),
        "length_penalty": float(cfg.get("length_penalty", 1.0)),
    }
    print(f"SIMP_LENGTH min_length={gen_kwargs['min_length']} "
          f"max_length={gen_kwargs['max_length']} "
          f"length_penalty={gen_kwargs['length_penalty']}", flush=True)

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
