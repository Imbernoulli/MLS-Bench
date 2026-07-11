#!/usr/bin/env python3
"""mt-sampling-vs-beam harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model, choosing HOW
to search: the agent returns a decode MODE (solution/sampling.py -> build_mode ->
one of the allowed strings). Scores corpus sacreBLEU / chrF.

For a peaked, well-trained MT model, MAP-style search (beam) beats stochastic
sampling on BLEU (Holtzman et al. 2020 note sampling helps open-ended generation,
NOT MT, where beam is the standard). Pure ancestral sampling (temperature 1.0) is
the weakest; nucleus (top-p) sampling recovers some quality; greedy is close to
low-temperature sampling; a real beam-5 decode is best.

Modes:
  "sample_t1"   : ancestral sampling, temperature 1.0     (weakest — high variance)
  "topp"        : nucleus sampling, top_p=0.9, temp 0.7   (better than pure sample)
  "greedy"      : greedy decoding (num_beams=1)           (deterministic argmax)
  "beam"        : beam-5 search, length_penalty 1.0       (strong — MAP search)

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

_VALID = {"sample_t1", "topp", "greedy", "beam"}
TASK_NAME = "mt-sampling-vs-beam"
SURFACE_NAME = "build_mode"

_CFG = {
    "sample_t1": {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                  "num_beams": 1, "max_new_tokens": 128},
    "topp":      {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 0,
                  "num_beams": 1, "max_new_tokens": 128},
    "greedy":    {"num_beams": 1, "max_new_tokens": 128},
    "beam":      {"num_beams": 5, "length_penalty": 1.0, "early_stopping": True,
                  "max_new_tokens": 128},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()
    common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)

    srcs, refs, data_proof = common.load_dataset()

    mode = common.load_surface_value(args.solution, "build_mode")
    if not isinstance(mode, str):
        raise TypeError("build_mode must return a string")
    if mode not in _VALID:
        raise SystemExit(f"mode must be one of {sorted(_VALID)} (got {mode!r})")
    model, tok, model_proof = common.load_model_and_tokenizer(dev)
    common.emit_provenance(model_proof, data_proof)
    preds = common.translate(model, tok, srcs, dict(_CFG[mode]), dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    common.emit_result(TASK_NAME, SURFACE_NAME, scores, plen, dt, len(srcs))


if __name__ == "__main__":
    main()
