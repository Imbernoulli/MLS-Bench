#!/usr/bin/env python3
"""simp-min-length harness (fixed pipeline; ISOLATED min-length floor).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using a FIXED beam width, length_penalty and
max_length (num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, max_length=96,
early_stopping=True all FIXED) and the agent's decoder-side min_length FLOOR
(solution/minlen.py -> build_min_length -> int). Scores corpus SARI per setting.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-minlen-floor"
SURFACE = "minlen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_min_length = common.load_surface(args.solution, "build_min_length")
    ml = common.require_surface_int(
        build_min_length(), "min_length", 0, 96, surface="build_min_length"
    )
    gen_kwargs = {
        "num_beams": 5,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "length_penalty": 1.0,          # FIXED
        "max_length": 96,               # FIXED
        "early_stopping": True,         # FIXED
        "min_length": ml,
    }
    print(f"SIMP_MINLEN min_length={gen_kwargs['min_length']}", flush=True)

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
