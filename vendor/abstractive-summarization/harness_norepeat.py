#!/usr/bin/env python3
"""summ-norepeat-ngram harness (3-setting).

The no-repeat n-gram size alone is editable; other decode controls are fixed.
the agent chooses only no_repeat_ngram_size (solution/norepeat.py ->
build_norepeat_size -> int; 0 disables). Neural summarizers on multi-sentence
outputs are prone to n-gram repetition loops; the classic 3-gram block removes
them and lifts ROUGE (biggest effect on the multi-sentence CNN/DM setting).

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

    build_norepeat_size = common.load_surface(args.solution, "build_norepeat_size")
    nrg = common.require_surface_int(
        build_norepeat_size(), "no_repeat_ngram_size", 0, 20,
        surface="build_norepeat_size",
    )
    print(f"SUMM_NOREPEAT no_repeat_ngram_size={nrg}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        beams = 6 if setting == "xsum" else 4
        return {
            "num_beams": beams,      # FIXED
            "early_stopping": True,
            "no_repeat_ngram_size": nrg,
            **win,
        }

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.time() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
