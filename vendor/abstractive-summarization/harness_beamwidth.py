#!/usr/bin/env python3
"""summ-beam-width harness (3-setting).

Beam width alone: the per-domain length and repetition controls are fixed; the
agent chooses only num_beams (solution/beamwidth.py -> build_beam_width -> int).
returned width is validated and scored with corpus ROUGE-L F1 on all settings.

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

    build_beam_width = common.load_surface(args.solution, "build_beam_width")
    beams = common.require_surface_int(
        build_beam_width(), "num_beams", 1, 12, surface="build_beam_width"
    )
    print(f"SUMM_BEAMWIDTH num_beams={beams}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        gk = {
            "num_beams": beams,
            "no_repeat_ngram_size": 3,  # FIXED
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
