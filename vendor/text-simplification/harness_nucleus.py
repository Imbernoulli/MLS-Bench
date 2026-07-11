#!/usr/bin/env python3
"""simp-nucleus-topp harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using SAMPLING (do_sample=True, num_beams=1,
temperature=1.0 all FIXED) restricted to the agent's NUCLEUS (top-p)
(solution/nucleus.py -> build_top_p -> float). Scores corpus SARI per setting.

Nucleus (top-p) sampling truncates the sampling distribution to the smallest set
of tokens whose cumulative probability is at least p, then samples from that
renormalized set. This harness fixes the remaining generation settings so only
top-p is visible.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-nucleus-sampling"
SURFACE = "nucleus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_top_p = common.load_surface(args.solution, "build_top_p")
    top_p = common.require_surface_number(
        build_top_p(), "top_p", 0.0, 1.0,
        surface="build_top_p", low_open=True,
    )
    gen_kwargs = {
        "do_sample": True,           # FIXED (sampling, not beam search)
        "num_beams": 1,              # FIXED
        "temperature": 1.0,          # FIXED
        "no_repeat_ngram_size": 3,   # FIXED
        "max_length": 128,           # FIXED
        "top_p": top_p,
    }
    print(f"SIMP_NUCLEUS top_p={gen_kwargs['top_p']}", flush=True)

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
