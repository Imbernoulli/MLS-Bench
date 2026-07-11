#!/usr/bin/env python3
"""summ-beam-repetition harness (3-setting).

Decodes EACH of the THREE FIXED domain settings (xsum / cnndm / samsum) with the
FROZEN domain-matched summarizer, per-domain length window FIXED, using the agent's
BEAM / REPETITION config (solution/beam.py -> build_beam_config -> {num_beams,
no_repeat_ngram_size, repetition_penalty}). Scores corpus ROUGE-L F1.

Emits one line per setting:
    SUMM_METRICS setting=<S> rougeL=<F> rouge1=<F> rouge2=<F> plen=<W>
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

    build_beam_config = common.load_surface(args.solution, "build_beam_config")
    cfg = common.require_surface_config(
        build_beam_config(),
        {"num_beams", "no_repeat_ngram_size", "repetition_penalty"},
        surface="build_beam_config",
    )
    print(f"SUMM_BEAM num_beams={cfg['num_beams']} "
          f"no_repeat_ngram_size={cfg['no_repeat_ngram_size']} "
          f"repetition_penalty={cfg['repetition_penalty']}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        gk = {
            "num_beams": cfg["num_beams"],
            "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
            "repetition_penalty": cfg["repetition_penalty"],
            **win,
        }
        if gk["num_beams"] > 1:
            gk["early_stopping"] = True
        return gk

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.time() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
