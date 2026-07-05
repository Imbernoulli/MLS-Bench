#!/usr/bin/env python3
"""simp-decoding-beam harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's BEAM / REPETITION decode
config (solution/beam.py -> build_beam_config -> {num_beams, no_repeat_ngram_size,
repetition_penalty}). The length window (max_length / length_penalty) is FIXED, so
ONLY the beam/repetition config varies. Scores corpus SARI per setting.

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

    build_beam_config = common.load_surface(args.solution, "build_beam_config")
    cfg = build_beam_config()
    gen_kwargs = {
        "num_beams": int(cfg.get("num_beams", 1)),
        "no_repeat_ngram_size": int(cfg.get("no_repeat_ngram_size", 0)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
        "max_length": 128,           # FIXED length window
        "length_penalty": 1.0,       # FIXED
        "early_stopping": True,      # FIXED
    }
    print(f"SIMP_BEAM num_beams={gen_kwargs['num_beams']} "
          f"no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']} "
          f"repetition_penalty={gen_kwargs['repetition_penalty']}", flush=True)

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
