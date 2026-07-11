#!/usr/bin/env python3
"""summ-sampling-vs-beam harness (3-setting).

Decoding STRATEGY: sampling vs beam. Per-domain length windows are fixed; the
agent returns a complete strategy mapping from solution/sampling.py via
build_decode_strategy. Beam mappings require num_beams, while sampling mappings
require top_p, top_k, and temperature. The harness validates the selected schema
and measures its multi-domain corpus ROUGE-L F1 without publishing an ordering.

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

    build_decode_strategy = common.load_surface(args.solution, "build_decode_strategy")
    cfg = common.require_surface_config(
        build_decode_strategy(), {"strategy"},
        allowed={"strategy", "num_beams", "top_p", "top_k", "temperature"},
        surface="build_decode_strategy",
    )
    strat = common.require_surface_choice(
        cfg["strategy"], "strategy", {"sample", "beam"},
        surface="build_decode_strategy",
    )
    if strat == "sample":
        common.require_surface_config(
            cfg, {"strategy", "top_p", "top_k", "temperature"},
            allowed={"strategy", "top_p", "top_k", "temperature"},
            surface="sample strategy",
        )
    elif strat == "beam":
        common.require_surface_config(
            cfg, {"strategy", "num_beams"},
            allowed={"strategy", "num_beams"}, surface="beam strategy",
        )
    print(f"SUMM_STRATEGY strategy={strat} "
          f"num_beams={cfg.get('num_beams')} top_p={cfg.get('top_p')} "
          f"top_k={cfg.get('top_k')} temperature={cfg.get('temperature')}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        gk = {"no_repeat_ngram_size": 3, **win}
        if strat == "sample":
            gk.update({
                "do_sample": True,
                "num_beams": 1,
                "top_p": cfg["top_p"],
                "top_k": cfg["top_k"],
                "temperature": cfg["temperature"],
            })
        else:
            gk.update({
                "do_sample": False,
                "num_beams": cfg["num_beams"],
                "early_stopping": True,
            })
        return gk

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
