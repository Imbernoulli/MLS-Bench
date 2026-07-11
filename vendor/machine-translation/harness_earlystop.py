#!/usr/bin/env python3
"""mt-early-stopping harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under beam-5
with a SHORT-biased length policy (length_penalty=0.6, the tuned MT optimum for
this model); the agent controls ONLY the beam-search STOPPING policy
(solution/earlystop.py -> build_early_stopping -> one of True / False / "never").
Scores corpus sacreBLEU / chrF.

`early_stopping` governs when beam search stops expanding (HF `generate`):
  True     : stop as soon as `num_beams` finished hypotheses exist. With a
             short-biased length_penalty this can stop too early on a still-
             improving beam -> slightly worse.
  False    : heuristic stop (stop when it is unlikely a better hypothesis remains).
  "never"  : canonical stopping — only stop when NO better hypothesis can exist
             given the length_penalty (Huang et al. 2017 "When to Finish?"); this
             is the theoretically-correct policy and matches/exceeds the others.

The gap is small (this is a genuine minor lever): "never" >= False >= True.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = (True, False, "never")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    raw = common.load_surface_value(args.solution, "build_early_stopping")
    if raw not in _VALID or not isinstance(raw, (bool, str)):
        raise SystemExit(f"early_stopping must be True/False/'never' (got {raw!r})")
    es = raw
    print(f"MT_EARLYSTOP early_stopping={es!r}", flush=True)

    gen_kwargs = {"num_beams": 5, "length_penalty": 0.6,
                  "early_stopping": es, "max_new_tokens": 128}
    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
