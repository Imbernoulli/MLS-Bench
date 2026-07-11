#!/usr/bin/env python3
"""summ-decoding-length harness (3-setting).

Decodes EACH of the THREE FIXED domain settings (xsum / cnndm / samsum) with the
FROZEN domain-matched summarizer, beam+no-repeat-3gram FIXED, using the agent's
LENGTH-CONTROL config (solution/length.py -> build_length_config -> {min_length,
max_length, length_penalty}) applied to all settings. Uses mean per-example ROUGE-L F1.

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
    t0 = time.perf_counter()

    build_length_config = common.load_surface(args.solution, "build_length_config")
    cfg = common.require_surface_config(
        build_length_config(),
        {"min_length", "max_length", "length_penalty"},
        surface="build_length_config",
    )
    minimum = common.require_surface_int(
        cfg["min_length"], "min_length", 0, 200, surface="build_length_config"
    )
    maximum = common.require_surface_int(
        cfg["max_length"], "max_length", 1, 200, surface="build_length_config"
    )
    penalty = common.require_surface_number(
        cfg["length_penalty"], "length_penalty", 0.0, 10.0,
        low_open=True, surface="build_length_config",
    )
    if minimum > maximum:
        print("SURFACE_ERROR build_length_config: min_length exceeds max_length",
              flush=True)
        raise ValueError("min_length exceeds max_length")
    print(f"SUMM_LENGTH min_length={minimum} max_length={maximum} "
          f"length_penalty={penalty}", flush=True)

    def build_gen(setting):
        beams = 6 if setting == "xsum" else 4
        return {
            "num_beams": beams,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "min_length": minimum,
            "max_length": maximum,
            "length_penalty": penalty,
        }

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
