#!/usr/bin/env python3
"""simp-model-capacity harness (fixed pipeline; WHICH FROZEN MODEL lever).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with the
agent's choice of FROZEN pretrained simplifier (solution/capacity.py ->
build_model_choice -> a string), using a FIXED beam decode config
(num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0) identical for every
choice, so ONLY the model checkpoint varies. Scores corpus SARI per setting.

Three frozen, staged-offline seq2seq checkpoints are supported:
  "small_turk"
  "small_wikiauto"
  "base_turk"

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-model-capacity"
SURFACE = "capacity"

_GEN = {
    "num_beams": 5,                 # FIXED, identical for every model
    "no_repeat_ngram_size": 3,      # FIXED
    "length_penalty": 1.0,          # FIXED
    "max_length": 128,              # FIXED
    "early_stopping": True,         # FIXED
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_choice = common.load_surface(args.solution, "build_model_choice")
    choice = common.require_surface_choice(
        build_choice(), "model_choice", set(common.MODEL_SPECS),
        surface="build_model_choice",
    )
    mpath, _ = common.model_identity(choice)
    print(f"SIMP_CAPACITY model_choice={choice} path={mpath}", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(mpath, local_files_only=True, torch_dtype=torch.float32)
    model.to(dev)
    model.eval()

    metric_lines = []
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = common.simplify(model, tok, srcs, dict(_GEN), dev)
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
        model_choice=choice, metric_lines=metric_lines,
        elapsed=time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
