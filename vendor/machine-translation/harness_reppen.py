#!/usr/bin/env python3
"""mt-repetition-penalty harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model using a FIXED
beam width (5) and length policy; the agent controls ONLY the `repetition_penalty`
(solution/reppen.py -> build_reppen_config -> {repetition_penalty}). Scores
corpus sacreBLEU / chrF.

`repetition_penalty` (Keskar et al. 2019 CTRL) divides the logits of already-
generated tokens by the penalty (>1.0 discourages repetition; 1.0 = off). A mild
penalty (~1.1-1.2) suppresses over-generation loops and helps a little; too large
(>=1.6) distorts fluent output and hurts.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

TASK_NAME = "mt-repetition-penalty"
SURFACE_NAME = "build_reppen_config"


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
        common.load_surface_value(args.solution, "build_reppen_config"),
        "build_reppen_config",
        {"repetition_penalty"},
    )
    rp = common.require_real(
        cfg["repetition_penalty"], "repetition_penalty", 0.1, 5.0
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "length_penalty": 1.0,          # FIXED
        "early_stopping": True,         # FIXED
        "max_new_tokens": 128,          # FIXED
        "repetition_penalty": rp,
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
