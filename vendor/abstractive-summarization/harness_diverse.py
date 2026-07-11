#!/usr/bin/env python3
"""summ-diverse-beam harness (3-setting).

Beam grouping is evaluated with the same repetition and per-domain length controls.
The agent returns a complete mapping from solution/diverse.py via
`build_diverse_config`. The mapping must contain beam width, beam-group count,
and diversity penalty, with groups dividing the beam count.
The harness does not publish a measured ordering between valid mappings.
It scores committed summaries with mean per-example ROUGE-L F1.
Invalid configurations fail before generation.

Emits one line per setting:
    SUMM_METRICS setting=<S> rougeL=<F> rouge1=<F> rouge2=<F> plen=<W>
"""
from __future__ import annotations

import argparse
import time

import common


def _validate_diverse_config(value) -> tuple[int, int, float]:
    """Validate the editable mapping before emitting proof or loading a model."""
    cfg = common.require_surface_config(
        value,
        {"num_beams", "num_beam_groups", "diversity_penalty"},
        surface="build_diverse_config",
    )
    nb = common.require_surface_int(
        cfg["num_beams"], "num_beams", 1, 12, surface="build_diverse_config"
    )
    ng = common.require_surface_int(
        cfg["num_beam_groups"], "num_beam_groups", 1, 12,
        surface="build_diverse_config",
    )
    dp = common.require_surface_number(
        cfg["diversity_penalty"], "diversity_penalty", 0.0, 10.0,
        surface="build_diverse_config",
    )
    if ng > nb or nb % ng:
        print("SURFACE_ERROR build_diverse_config: beam groups must divide num_beams",
              flush=True)
        raise ValueError("invalid beam grouping")
    if (ng == 1 and dp != 0.0) or (ng > 1 and dp <= 0.0):
        print(
            "SURFACE_ERROR build_diverse_config: diversity_penalty must be 0 "
            "for one group and >0 for grouped beams",
            flush=True,
        )
        raise ValueError("invalid diversity penalty for beam grouping")
    return nb, ng, dp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_diverse_config = common.load_surface(args.solution, "build_diverse_config")
    nb, ng, dp = _validate_diverse_config(build_diverse_config())
    print(f"SUMM_DIVERSE num_beams={nb} num_beam_groups={ng} "
          f"diversity_penalty={dp}", flush=True)

    def build_gen(setting):
        win = common.LEN_WINDOW[setting]
        gk = {
            "num_beams": nb,
            "num_beam_groups": ng,
            "no_repeat_ngram_size": 3,
            **win,
        }
        if ng > 1:
            gk["diversity_penalty"] = dp
        else:
            gk["early_stopping"] = True
        return gk

    common.run_over_settings(build_gen, dev)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
