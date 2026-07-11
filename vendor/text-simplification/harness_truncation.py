#!/usr/bin/env python3
"""simp-input-truncation harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier under a FIXED beam decode
(num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, max_length=128), varying
ONLY the agent's SOURCE-SIDE INPUT TRUNCATION budget (solution/truncation.py ->
build_max_input_tokens -> int; the tokenizer's `max_length` / `truncation=True` on
the ENCODER side). Scores corpus SARI per setting.

The encoder input is truncated to `max_input_tokens` tokens, hard-capped by the
harness. This isolates the encoder-side truncation lever from the fixed decoder
configuration.

Emits three task-bound v2 metric records and one unique terminal SIMP_DONE proof.
"""
from __future__ import annotations

import argparse
import time

import common

TASK_ID = "simp-input-truncation"
SURFACE = "truncation"

_GEN = {
    "num_beams": 5,                 # FIXED
    "no_repeat_ngram_size": 3,      # FIXED
    "length_penalty": 1.0,          # FIXED
    "max_length": 128,              # FIXED (decoder side; unrelated to input budget)
    "early_stopping": True,         # FIXED
}


def _simplify_with_input_budget(model, tok, sources, max_input_tokens, gen_kwargs, device):
    import torch

    if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int):
        print("SURFACE_ERROR build_max_input_tokens must return an integer", flush=True)
        raise TypeError("max input tokens must be an integer")
    if not 8 <= max_input_tokens <= common.MAX_INPUT_TOKENS:
        print(f"SURFACE_ERROR max_input_tokens outside [8, "
              f"{common.MAX_INPUT_TOKENS}]: {max_input_tokens}", flush=True)
        raise ValueError("max input tokens outside allowed range")
    gk = common._sanitize_gen_kwargs(gen_kwargs)
    preds = []
    for i in range(0, len(sources), common.GEN_BATCH_SIZE):
        batch = [common.SRC_PREFIX + s for s in sources[i:i + common.GEN_BATCH_SIZE]]
        enc = tok(
            batch,
            max_length=max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **gk,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    return [p.strip() for p in preds], max_input_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_mit = common.load_surface(args.solution, "build_max_input_tokens")
    mit = build_mit()

    model, tok = common.load_model_and_tokenizer(dev)
    metric_lines = []
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds, used_mit = _simplify_with_input_budget(model, tok, srcs, mit, dict(_GEN), dev)
        if setting == common.SETTINGS[0]:
            print(f"SIMP_TRUNCATION max_input_tokens={used_mit}", flush=True)
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
