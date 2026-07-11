#!/usr/bin/env python3
"""mt-length-penalty harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with the
matching FROZEN OPUS-MT MarianMT model, using the agent's LENGTH-NORMALIZATION
coefficient (solution/length.py -> build_length_config -> {length_penalty}).
Beam width, minimum length, and output budget are fixed so only normalization
varies.
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

TASK_NAME = "mt-length-penalty"
SURFACE_NAME = "build_length_config"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()
    common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)

    srcs, refs, data_proof = common.load_dataset()

    cfg = common.require_config(
        common.load_surface_value(args.solution, "build_length_config"),
        "build_length_config",
        {"length_penalty"},
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 0,      # FIXED (MT rarely loops; keep it clean)
        "early_stopping": True,         # FIXED
        "length_penalty": common.require_real(
            cfg["length_penalty"], "length_penalty", 0.0, 5.0
        ),
        "min_length": 0,                   # FIXED
        "max_new_tokens": 128,             # FIXED; owned by mt-batch-maxlen
    }
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
