#!/usr/bin/env python3
"""summ-nucleus-topp harness (3-setting).

Nucleus (top-p) CUTOFF under sampling: sampling is ON (do_sample) with a FIXED
temperature, repetition, and per-domain length controls are fixed; the agent chooses
only top_p (solution/topp.py -> build_top_p -> float). The cutoff changes the
candidate probability mass while all other controls remain constant. The harness
measures complete multi-domain corpus ROUGE-L F1 without publishing a preferred
value or measured ordering.

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

    build_top_p = common.load_surface(args.solution, "build_top_p")
    tp = common.require_surface_number(
        build_top_p(), "top_p", 0.05, 1.0, surface="build_top_p"
    )
    print(f"SUMM_TOPP top_p={tp}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        return {
            "do_sample": True,
            "num_beams": 1,
            "top_p": tp,          # varies
            "top_k": 0,
            "temperature": 1.0,   # FIXED
            "no_repeat_ngram_size": 3,
            **win,
        }

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
