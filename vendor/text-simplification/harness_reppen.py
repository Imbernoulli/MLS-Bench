#!/usr/bin/env python3
"""simp-repetition-penalty harness (fixed pipeline; ISOLATED from beam width).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, GREEDY decode (num_beams=1 FIXED, so the
lever is isolated from beam search), at the agent's REPETITION PENALTY
(solution/reppen.py -> build_repetition_penalty -> float). Scores corpus SARI per
setting.

repetition_penalty (Keskar et al. 2019 CTRL-style) down-weights previously
generated tokens' logits (>1.0 discourages repeats, 1.0 = off). Isolated here under
GREEDY decode (num_beams=1, no_repeat_ngram_size=0) so its effect is visible without
beam search or n-gram blocking masking it: greedy T5 simplifiers can loop on a
frequent function word without a repetition penalty; the loop wastes generation
budget and rarely lands on the correct simplified phrasing, hurting SARI's ADD/KEEP
credit. A moderate penalty (~1.2-1.5) breaks loops and is the strong setting here.

Emits one metric line PER SETTING:
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>
"""
from __future__ import annotations

import argparse
import time

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    build_rp = common.load_surface(args.solution, "build_repetition_penalty")
    rp = float(build_rp())
    gen_kwargs = {
        "num_beams": 1,                 # FIXED (greedy — isolates from beam search)
        "no_repeat_ngram_size": 0,      # FIXED (isolates from n-gram blocking)
        "max_length": 128,              # FIXED
        "repetition_penalty": rp,
    }
    print(f"SIMP_REPPEN repetition_penalty={gen_kwargs['repetition_penalty']}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    for setting in common.SETTINGS:
        srcs, refs = common.load_dataset(setting)
        preds = common.simplify(model, tok, srcs, dict(gen_kwargs), dev)
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
