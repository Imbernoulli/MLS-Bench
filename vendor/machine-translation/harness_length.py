#!/usr/bin/env python3
"""mt-length-penalty harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with the
matching FROZEN OPUS-MT MarianMT model, using the agent's LENGTH-NORMALIZATION decode
config (solution/length.py -> build_length_config -> {length_penalty, min_length,
max_new_tokens}). num_beams is FIXED to 5 here so ONLY the length policy varies.
Scores corpus sacreBLEU (and chrF) on the FIXED references.

Beam search WITHOUT length normalization biases toward too-short translations
whose BLEU is dragged down by the brevity penalty (Wu et al. 2016 GNMT length
penalty); a tuned length_penalty recovers it. In HF `generate`, the beam score is
divided by length**length_penalty, so length_penalty>1 promotes longer, <1
shorter, 1.0 is plain mean-log-prob normalization; the DEFAULT here is the WEAK
short-biased length_penalty.

Emits one metric line:
    MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
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
        common.load_surface_value(args.solution, "build_length_config"),
        "build_length_config",
        {"length_penalty", "min_length", "max_new_tokens"},
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 0,      # FIXED (MT rarely loops; keep it clean)
        "early_stopping": True,         # FIXED
        "length_penalty": common.require_real(
            cfg["length_penalty"], "length_penalty", 0.0, 5.0
        ),
        "min_length": common.require_int(
            cfg["min_length"], "min_length", 0, common.MAX_NEW_TOKENS_CAP
        ),
        "max_new_tokens": common.require_int(
            cfg["max_new_tokens"], "max_new_tokens", 1, common.MAX_NEW_TOKENS_CAP
        ),
    }
    if gen_kwargs["min_length"] > gen_kwargs["max_new_tokens"]:
        raise ValueError("min_length cannot exceed max_new_tokens")
    print(f"MT_LENGTH length_penalty={gen_kwargs['length_penalty']} "
          f"min_length={gen_kwargs['min_length']} "
          f"max_new_tokens={gen_kwargs['max_new_tokens']}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
