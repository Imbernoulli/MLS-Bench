#!/usr/bin/env python3
"""summ-decoding-length harness (3-setting).

Decodes EACH of the THREE FIXED domain settings (xsum / cnndm / samsum) with the
FROZEN domain-matched summarizer, beam+no-repeat-3gram FIXED, using the agent's
LENGTH-CONTROL config (solution/length.py -> build_length_config -> {min_length,
max_length, length_penalty}) applied to all settings. Scores corpus ROUGE-L F1.

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

    build_length_config = common.load_surface(args.solution, "build_length_config")
    cfg = common.require_surface_config(
        build_length_config(),
        {"min_length", "max_length", "length_penalty"},
        surface="build_length_config",
    )
    print(f"SUMM_LENGTH min_length={cfg['min_length']} "
          f"max_length={cfg['max_length']} "
          f"length_penalty={cfg['length_penalty']}", flush=True)

    def build_gen(setting):
        beams = 6 if setting == "xsum" else 4
        return {
            "num_beams": beams,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "min_length": cfg["min_length"],
            "max_length": cfg["max_length"],
            "length_penalty": cfg["length_penalty"],
        }

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.time() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
