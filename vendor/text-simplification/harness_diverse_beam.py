#!/usr/bin/env python3
"""simp-diverse-beam harness (fixed pipeline).

Simplifies EACH of the THREE FIXED test settings (asset / turk / wiki) with a
FROZEN pretrained t5-base simplifier, using DIVERSE BEAM SEARCH / GROUP BEAM
SEARCH at the agent's group config (solution/diverse_beam.py ->
build_diverse_beam_config -> {num_beam_groups, diversity_penalty}). num_beams=6 and
no_repeat_ngram_size=3 are FIXED, so ONLY the grouping / diversity-penalty config
varies. Scores corpus SARI per setting.

Diverse beam search (Vijayakumar et al. 2016) splits the beams into groups and
penalises later groups for repeating earlier groups' tokens (diversity_penalty),
trading beam-1-best sharpness for hypothesis diversity. num_beam_groups=1 (the
FIXED-hypothesis-count degenerate case, diversity_penalty inert) recovers plain beam
search; a HIGH diversity penalty with many groups pushes hypotheses apart at the
cost of the single best (top-1) sequence quality that greedy/beam-decode metrics
like SARI reward — the plain (non-diverse) beam decode is the strong setting here.

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

    build_cfg = common.load_surface(args.solution, "build_diverse_beam_config")
    cfg = build_cfg()
    num_beam_groups = int(cfg.get("num_beam_groups", 1))
    diversity_penalty = float(cfg.get("diversity_penalty", 0.0))
    gen_kwargs = {
        "num_beams": 6,                 # FIXED
        "no_repeat_ngram_size": 3,      # FIXED
        "max_length": 128,              # FIXED
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity_penalty,
    }
    print(f"SIMP_DIVERSE_BEAM num_beam_groups={num_beam_groups} "
          f"diversity_penalty={diversity_penalty}", flush=True)

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
