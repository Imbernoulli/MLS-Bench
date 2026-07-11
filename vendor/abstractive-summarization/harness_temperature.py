#!/usr/bin/env python3
"""summ-decoding-temperature harness (3-setting).

Sampling temperature is editable while nucleus, length, and repetition controls
remain fixed; the agent chooses only the temperature
(solution/temperature.py -> build_temperature -> float). Temperature changes the
sampling distribution while all other decode controls remain constant. The
harness measures the complete multi-domain corpus ROUGE-L F1 without publishing
a preferred value or measured ordering.

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

    build_temperature = common.load_surface(args.solution, "build_temperature")
    temp = common.require_surface_number(
        build_temperature(), "temperature", 0.05, 5.0,
        surface="build_temperature",
    )
    print(f"SUMM_TEMPERATURE temperature={temp}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        return {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.95,       # FIXED nucleus
            "top_k": 0,
            "temperature": temp,
            "no_repeat_ngram_size": 3,
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
