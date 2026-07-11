#!/usr/bin/env python3
"""simp-length-control harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's LENGTH / COMPRESSION decode
config (solution/length.py -> build_length_config -> {min_length, max_length,
length_penalty}). num_beams and no_repeat_ngram_size are FIXED, so ONLY the length
window varies. Scores corpus SARI per setting.

Length settings affect the output/input balance measured by SARI. This harness
fixes the remaining generation settings so only the length controls vary.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-length-control"
SURFACE = "length"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_length_config = common.load_surface(args.solution, "build_length_config")
    cfg = common.require_surface_config(
        build_length_config(),
        {"min_length", "max_length", "length_penalty"},
        surface="build_length_config",
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "early_stopping": True,         # FIXED
        "min_length": cfg["min_length"],
        "max_length": cfg["max_length"],
        "length_penalty": cfg["length_penalty"],
    }
    print(f"SIMP_LENGTH min_length={gen_kwargs['min_length']} "
          f"max_length={gen_kwargs['max_length']} "
          f"length_penalty={gen_kwargs['length_penalty']}", flush=True)

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
