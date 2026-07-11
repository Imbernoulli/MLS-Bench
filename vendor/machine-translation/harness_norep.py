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

TASK_NAME = "mt-no-repeat-ngram"
SURFACE_NAME = "build_norep_config"


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
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
