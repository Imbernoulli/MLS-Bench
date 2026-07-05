#!/usr/bin/env python3
"""simp-input-truncation harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier under a FIXED strong beam decode
(num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, max_length=128), varying
ONLY the agent's SOURCE-SIDE INPUT TRUNCATION budget (solution/truncation.py ->
build_max_input_tokens -> int; the tokenizer's `max_length` / `truncation=True` on
the ENCODER side). Scores corpus SARI per setting.

The encoder input is truncated to `max_input_tokens` tokens (hard-capped in
[8, 160]). Text-simplification sources are short sentences (mean ~15-25 words for
asset/turk; WikiAuto sources run longer, up to 80 words = 100+ subword tokens): an
AGGRESSIVELY SHORT input budget silently drops the tail of longer sources before
the model ever sees it, and the model then has no way to recover the deleted
content's ADD/KEEP credit — SARI drops, especially on the longer `wiki` setting. A
generous budget (the model's real max, ~160 tokens) lets every source be read in
full. This isolates the ENCODER-side truncation lever from the (FIXED) decode
config used by every other simp-* task.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common

_GEN = {
    "num_beams": 5,                 # FIXED
    "no_repeat_ngram_size": 3,      # FIXED
    "length_penalty": 1.0,          # FIXED
    "max_length": 128,              # FIXED (decoder side; unrelated to input budget)
    "early_stopping": True,         # FIXED
}


def _simplify_with_input_budget(model, tok, sources, max_input_tokens, gen_kwargs, device):
    import torch

    max_input_tokens = int(min(max(int(max_input_tokens), 8), common.MAX_INPUT_TOKENS))
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
    t0 = time.time()

    build_mit = common.load_surface(args.solution, "build_max_input_tokens")
    mit = int(build_mit())

    model, tok = common.load_model_and_tokenizer(dev)
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds, used_mit = _simplify_with_input_budget(model, tok, srcs, mit, dict(_GEN), dev)
        if setting == common.SETTINGS[0]:
            print(f"SIMP_TRUNCATION max_input_tokens={used_mit}", flush=True)
        sari = common.score_sari(srcs, preds, refs)
        bleu = common.bleu_corpus(preds, refs)
        plen = common.mean_pred_len_words(preds)
        lr = common.length_ratio(srcs, preds)
        print(f"SIMP_METRICS setting={setting} sari={sari:.6f} bleu={bleu:.4f} "
              f"n_sents={len(srcs)} plen={plen:.1f} lenratio={lr:.3f}", flush=True)

    dt = time.time() - t0
    print(f"SIMP_DONE elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
