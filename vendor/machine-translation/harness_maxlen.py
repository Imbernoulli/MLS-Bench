#!/usr/bin/env python3
"""mt-batch-maxlen harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under a FIXED
beam-5 length-1.0 decode; the agent controls ONLY the output-length budget
`max_new_tokens` (solution/maxlen.py -> build_max_new_tokens -> int). Scores
corpus sacreBLEU / chrF.

A too-tight generation budget (e.g. 8-16 new tokens) TRUNCATES the translation ->
the tail of the English sentence is cut off -> the brevity penalty and lost
n-grams tank BLEU. A budget >= the natural target length (~64-128) lets the model
finish the sentence. This is the "did you give the decoder enough room to finish?"
lever. (Batch size is a throughput-only knob and is held fixed.)

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

TASK_NAME = "mt-batch-maxlen"
SURFACE_NAME = "build_max_new_tokens"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()
    common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)

    srcs, refs, data_proof = common.load_dataset()

    mnt = common.require_int(
        common.load_surface_value(args.solution, "build_max_new_tokens"),
        "max_new_tokens",
        1,
        common.MAX_NEW_TOKENS_CAP,
    )
    gen_kwargs = {"num_beams": 5, "length_penalty": 1.0,
                  "early_stopping": True, "max_new_tokens": mnt}
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
