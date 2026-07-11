#!/usr/bin/env python3
"""mt-diverse-beam harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model using DIVERSE
beam search (Vijayakumar et al. 2016, "Diverse Beam Search"): the agent controls
the number of beam GROUPS and the diversity penalty (solution/divbeam.py ->
build_divbeam_config -> {num_beam_groups, diversity_penalty}); num_beams is FIXED
at 8. Scores corpus sacreBLEU / chrF (single top hypothesis).

Diverse beam search partitions the 8 beams into groups and adds a diversity
penalty across groups. For single-best MT quality there is a trade-off: modest
grouping/diversity (2 groups, small penalty) keeps quality near plain beam, while
too many groups with a large penalty (8 groups, penalty>=1.0) forces the top
hypothesis off the high-probability path -> lower BLEU. Plain beam (1 group) is
the reference; over-diversified is the degenerate.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
"""
from __future__ import annotations

import argparse
import time

import common

_NUM_BEAMS = 8


def _generation_kwargs(cfg: dict) -> dict:
    """Validate the public surface before calling Transformers generation."""
    groups = common.require_int(
        cfg["num_beam_groups"], "num_beam_groups", 1, _NUM_BEAMS
    )
    if _NUM_BEAMS % groups:
        raise ValueError(f"num_beam_groups must divide {_NUM_BEAMS}, got {groups}")
    div = common.require_real(
        cfg["diversity_penalty"], "diversity_penalty", 0.0, 5.0
    )
    if groups == 1 and div != 0.0:
        raise ValueError("diversity_penalty must be zero when num_beam_groups is one")
    if groups > 1 and div <= 0.0:
        raise ValueError(
            "diversity_penalty must be strictly positive with multiple beam groups"
        )

    kwargs = {
        "num_beams": _NUM_BEAMS,
        "length_penalty": 1.0,
        "max_new_tokens": 128,
    }
    if groups > 1:
        kwargs["num_beam_groups"] = groups
        kwargs["diversity_penalty"] = div
    else:
        kwargs["early_stopping"] = True
    return kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.time()

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    cfg = common.require_config(
        common.load_surface_value(args.solution, "build_divbeam_config"),
        "build_divbeam_config",
        {"num_beam_groups", "diversity_penalty"},
    )
    gen_kwargs = _generation_kwargs(cfg)
    groups = int(cfg["num_beam_groups"])
    div = float(cfg["diversity_penalty"])
    print(f"MT_DIVBEAM num_beam_groups={groups} diversity_penalty={div}", flush=True)

    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
