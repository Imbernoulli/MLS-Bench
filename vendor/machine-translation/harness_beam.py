#!/usr/bin/env python3
"""mt-decoding-beam harness (fixed pipeline).

Translates a complete pinned OPUS-100 test split with the matching FROZEN
OPUS-MT MarianMT model, using the agent's BEAM / REPETITION decode config
(solution/beam.py -> build_beam_config -> {num_beams, no_repeat_ngram_size}). The
length policy (length_penalty / max_new_tokens) is FIXED to a sensible MT window
here so ONLY the beam / repetition control varies. Scores corpus sacreBLEU (and
chrF) on the FIXED references.

Emits one metric line:
    MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

TASK_NAME = "mt-decoding-beam"
SURFACE_NAME = "build_beam_config"


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
        common.load_surface_value(args.solution, "build_beam_config"),
        "build_beam_config",
        {"num_beams", "no_repeat_ngram_size"},
    )
    gen_kwargs = {
        "length_penalty": 1.0,          # FIXED sensible MT length policy
        "max_new_tokens": 128,          # FIXED
        "num_beams": common.require_int(cfg["num_beams"], "num_beams", 1, 12),
        "no_repeat_ngram_size": common.require_int(
            cfg["no_repeat_ngram_size"], "no_repeat_ngram_size", 0, 10
        ),
    }
    if gen_kwargs["num_beams"] > 1:
        gen_kwargs["early_stopping"] = True
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
