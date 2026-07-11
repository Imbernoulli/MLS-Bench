#!/usr/bin/env python3
"""mt-postprocess-detok harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under a FIXED
beam-5 length-1.0 decode, then the agent chooses a POST-PROCESSING / detok policy
applied to the model output before it is scored (solution/postproc.py ->
build_postproc -> a rule string). Scores corpus sacreBLEU / chrF against the
FIXED (cased, punctuated) English references.

sacreBLEU is applied to raw text with an internal tokenizer, so what matters is
whether the surface form MATCHES the reference convention. The model already emits
properly cased, punctuated, SentencePiece-detokenized English, so the correct
policy is a light NORMALIZATION (collapse whitespace) — anything close to identity.
Lossy "normalizations" that DROP information the references keep tank BLEU:
  "identity"   : model output unchanged                              [reference]
  "normalize"  : collapse repeated whitespace / strip edges          [strong; ~identity]
  "lowercase"  : lowercase everything -> mismatches cased references  [degenerate]
  "strip_punct": remove all punctuation -> loses ref punctuation n-grams [degenerate]

This is the "don't destroy the model's good detok with a bad post-processor" lever;
a lossy post-processor must score LOWER than leaving the fluent output alone.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import re
import time

import common

_VALID = {"identity", "normalize", "lowercase", "strip_punct"}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _apply(preds, rule):
    if rule == "identity":
        return preds
    if rule == "normalize":
        return [_WS.sub(" ", p).strip() for p in preds]
    if rule == "lowercase":
        return [_WS.sub(" ", p).strip().lower() for p in preds]
    # strip_punct
    return [_WS.sub(" ", _PUNCT.sub(" ", p)).strip() for p in preds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    rule = common.load_surface_value(args.solution, "build_postproc")
    if not isinstance(rule, str):
        raise TypeError("build_postproc must return a string")
    if rule not in _VALID:
        raise SystemExit(f"postproc must be one of {sorted(_VALID)} (got {rule!r})")
    print(f"MT_POSTPROC rule={rule}", flush=True)

    gen_kwargs = {"num_beams": 5, "length_penalty": 1.0,
                  "early_stopping": True, "max_new_tokens": 128}
    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    preds = _apply(preds, rule)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
