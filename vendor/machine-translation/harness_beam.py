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
    print(f"MT_BEAM num_beams={gen_kwargs['num_beams']} "
          f"no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
