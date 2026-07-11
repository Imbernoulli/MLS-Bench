#!/usr/bin/env python3
"""mt-early-stopping harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under beam-5
with a fixed short-biased length policy (length_penalty=0.6); the agent controls
ONLY the beam-search STOPPING policy
(solution/earlystop.py -> build_early_stopping -> one of True / False / "never").
Scores corpus sacreBLEU / chrF.

`early_stopping` governs when beam search stops expanding (HF `generate`):
  True     : stop as soon as `num_beams` finished hypotheses exist.
  False    : heuristic stop (stop when it is unlikely a better hypothesis remains).
  "never"  : canonical stopping — only stop when NO better hypothesis can exist
             given the length penalty (Huang et al. 2017 "When to Finish?").

These criteria can select different hypotheses because the fixed length policy
changes the bound used to decide whether unfinished beams can still improve.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = (True, False, "never")
TASK_NAME = "mt-early-stopping"
SURFACE_NAME = "build_early_stopping"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()
    common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)

    srcs, refs, data_proof = common.load_dataset()

    raw = common.load_surface_value(args.solution, "build_early_stopping")
    if raw not in _VALID or not isinstance(raw, (bool, str)):
        raise SystemExit(f"early_stopping must be True/False/'never' (got {raw!r})")
    es = raw
    gen_kwargs = {"num_beams": 5, "length_penalty": 0.6,
                  "early_stopping": es, "max_new_tokens": 128}
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
