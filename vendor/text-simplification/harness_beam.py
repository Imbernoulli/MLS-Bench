#!/usr/bin/env python3
"""simp-decoding-beam harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's BEAM / REPETITION decode
config (solution/beam.py -> build_beam_config -> {num_beams, no_repeat_ngram_size,
repetition_penalty}). The length window (max_length / length_penalty) is FIXED, so
ONLY the beam/repetition config varies. Scores corpus SARI per setting.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-decoding-beam"
SURFACE = "beam"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_beam_config = common.load_surface(args.solution, "build_beam_config")
    cfg = common.require_surface_config(
        build_beam_config(),
        {"num_beams", "no_repeat_ngram_size", "repetition_penalty"},
        surface="build_beam_config",
    )
    gen_kwargs = {
        "num_beams": cfg["num_beams"],
        "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
        "repetition_penalty": cfg["repetition_penalty"],
        "max_length": 128,           # FIXED length window
        "length_penalty": 1.0,       # FIXED
        "early_stopping": True,      # FIXED
    }
    print(f"SIMP_BEAM num_beams={gen_kwargs['num_beams']} "
          f"no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']} "
          f"repetition_penalty={gen_kwargs['repetition_penalty']}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    metric_lines = []
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = common.simplify(model, tok, srcs, dict(gen_kwargs), dev)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        metric_lines.append(common.emit_metrics(
            task=TASK_ID, surface=SURFACE, setting=setting, sari=sari,
            bleu=bleu, n_sents=len(srcs), plen=plen, lenratio=lr,
        ))

    common.emit_done(
        task=TASK_ID, surface=SURFACE, seed=args.seed,
        model_choice="base_turk", metric_lines=metric_lines,
        elapsed=time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
