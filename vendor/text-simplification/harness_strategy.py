#!/usr/bin/env python3
"""simp-sampling-vs-beam harness (fixed pipeline; decoding STRATEGY choice).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using the agent's DECODING STRATEGY
(solution/strategy.py -> build_strategy -> a string):

  "sample" : plain multinomial sampling.
  "topp"   : nucleus sampling.
  "beam"   : deterministic beam search.

All settings share the SAME fixed length window (max_length=128) and the SAME
frozen model; only the search strategy varies.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-decoding-strategy"
SURFACE = "strategy"

_VALID = {"sample", "topp", "beam"}

_GEN = {
    "sample": {"do_sample": True, "num_beams": 1, "temperature": 1.0, "max_length": 128},
    "topp": {"do_sample": True, "num_beams": 1, "top_p": 0.9, "temperature": 1.0, "max_length": 128},
    "beam": {"num_beams": 5, "no_repeat_ngram_size": 3, "max_length": 128},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_strategy = common.load_surface(args.solution, "build_strategy")
    strategy = common.require_surface_choice(
        build_strategy(), "strategy", _VALID, surface="build_strategy"
    )
    gen_kwargs = dict(_GEN[strategy])
    print(f"SIMP_STRATEGY strategy={strategy}", flush=True)

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
