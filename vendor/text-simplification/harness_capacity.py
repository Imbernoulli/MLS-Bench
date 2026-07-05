#!/usr/bin/env python3
"""simp-model-capacity harness (fixed pipeline; WHICH FROZEN MODEL lever).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with the
agent's choice of FROZEN pretrained simplifier (solution/capacity.py ->
build_model_choice -> a string), using a FIXED strong beam decode config
(num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0) identical for every
choice, so ONLY the model checkpoint varies. Scores corpus SARI per setting.

Three FROZEN, staged-offline seq2seq checkpoints (none trained/fine-tuned here —
this task selects among EXISTING community checkpoints, all originally trained on
subsets of the SAME wiki_auto_asset_turk family of corpora):
  "small_turk"    : t5-small-finetuned-turk-text-simplification (t5-small, 60M
                    params, fine-tuned mainly on TurkCorpus-style lexical edits).
  "small_wikiauto": t5-small-finetuned-text-simplification (t5-small, 60M params,
                    fine-tuned on the broader wiki_auto_asset_turk mix).
  "base_turk"     : t5-base-finetuned-turk-text-simplification (t5-base, ~220M
                    params, the model used by every OTHER simp-* task) — MORE
                    CAPACITY (t5-base vs t5-small) is the standard "bigger backbone
                    helps" lever, holding the fine-tuning recipe family + decode
                    config fixed.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common

_MODEL_DIRS = {
    "small_turk": "t5-small-finetuned-turk-text-simplification",
    "small_wikiauto": "t5-small-finetuned-text-simplification",
    "base_turk": "t5-base-finetuned-turk-text-simplification",
}

_GEN = {
    "num_beams": 5,                 # FIXED (strong beam, identical for every model)
    "no_repeat_ngram_size": 3,      # FIXED
    "length_penalty": 1.0,          # FIXED
    "max_length": 128,              # FIXED
    "early_stopping": True,         # FIXED
}


def _model_path_for(choice: str) -> str:
    import os
    from pathlib import Path

    if choice not in _MODEL_DIRS:
        raise SystemExit(f"model choice must be one of {sorted(_MODEL_DIRS)} (got {choice!r})")
    base = Path(common.model_path()).parent  # .../models/
    p = base / _MODEL_DIRS[choice]
    if not p.exists():
        raise SystemExit(f"model dir not staged: {p}")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    build_choice = common.load_surface(args.solution, "build_model_choice")
    choice = str(build_choice()).strip()
    mpath = _model_path_for(choice)
    print(f"SIMP_CAPACITY model_choice={choice} path={mpath}", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(mpath, local_files_only=True, torch_dtype=torch.float32)
    model.to(dev)
    model.eval()

    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = common.simplify(model, tok, srcs, dict(_GEN), dev)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        print(f"SIMP_METRICS setting={setting} sari={sari:.6f} bleu={bleu:.4f} "
              f"n_sents={len(srcs)} plen={plen:.1f} lenratio={lr:.3f}", flush=True)

    dt = time.time() - t0
    print(f"SIMP_DONE model_choice={choice} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
