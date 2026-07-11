#!/usr/bin/env python3
"""mt-postprocess-detok harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under a FIXED
beam-5 length-1.0 decode, then the agent chooses a POST-PROCESSING / detok policy
applied to the model output before it is scored (solution/postproc.py ->
build_postproc -> a rule string). Scores corpus sacreBLEU / chrF against the
FIXED (cased, punctuated) English references.

sacreBLEU is applied to raw text with an internal tokenizer, so the selected
policy determines which surface convention reaches the metric:
  "normalize"  : collapse repeated whitespace and strip edges
  "lowercase"  : normalize whitespace and lowercase the output
  "strip_punct": remove punctuation and normalize whitespace

The alternatives are evaluated as literal deterministic transformations; the
fixed corpus metric decides which surface convention is most compatible.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import re
import time

import common

_VALID = {"normalize", "lowercase", "strip_punct"}
TASK_NAME = "mt-postprocess-detok"
SURFACE_NAME = "build_postproc"
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _apply(preds, rule):
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
    common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)

    srcs, refs, data_proof = common.load_dataset()

    rule = common.load_surface_value(args.solution, "build_postproc")
    if not isinstance(rule, str):
        raise TypeError("build_postproc must return a string")
    if rule not in _VALID:
        raise SystemExit(f"postproc must be one of {sorted(_VALID)} (got {rule!r})")
    gen_kwargs = {"num_beams": 5, "length_penalty": 1.0,
                  "early_stopping": True, "max_new_tokens": 128}
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    preds = _apply(preds, rule)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
