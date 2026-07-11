#!/usr/bin/env python3
"""mt-decoding-temperature harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model in SAMPLING
mode (do_sample=True, top_p=0.95 fixed), with the RNG seed fixed for
reproducibility; the agent controls ONLY the softmax `temperature`
(solution/temperature.py -> build_temperature -> float). Scores corpus
sacreBLEU / chrF.

Temperature scales the logits before softmax. High temperature (>=1.0) flattens
the distribution -> more random, lower-BLEU samples; low temperature (->0)
sharpens toward the argmax (approaching greedy) -> higher BLEU for a peaked MT
model. There is a clear monotone benefit to LOWERING temperature here (MT is not
open-ended generation); the sweet spot is low (~0.3-0.5).

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
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

    temp = common.require_real(
        common.load_surface_value(args.solution, "build_temperature"),
        "temperature",
        0.05,
        5.0,
    )
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.95,                  # FIXED nucleus window
        "top_k": 0,                     # FIXED
        "num_beams": 1,                 # FIXED (sampling)
        "temperature": temp,
        "max_new_tokens": 128,
    }
    print(f"MT_TEMP temperature={temp}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
