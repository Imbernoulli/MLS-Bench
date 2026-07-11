#!/usr/bin/env python3
"""summ-decoding-temperature harness (3-setting).

Sampling temperature is editable while nucleus, length, and repetition controls
remain fixed; the agent chooses only the temperature
(solution/temperature.py -> build_temperature -> float). Temperature changes the
sampling distribution while all other decode controls remain constant. The
harness measures mean per-example ROUGE-L F1 without publishing
a preferred value or measured ordering.

Emits one line per setting:
    SUMM_METRICS setting=<S> rougeL=<F> rouge1=<F> rouge2=<F> plen=<W>
"""
from __future__ import annotations

import argparse
import time

import common


def _validate_temperature(value) -> float:
    """Validate the standalone temperature task's task-specific closed interval."""
    return common.require_surface_number(
        value, "temperature", 0.05, 5.0, surface="build_temperature"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_temperature = common.load_surface(args.solution, "build_temperature")
    temp = _validate_temperature(build_temperature())
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
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
