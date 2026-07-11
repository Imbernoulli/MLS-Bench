#!/usr/bin/env python3
"""mt-tokenization-truncation harness (fixed pipeline).

Translates each complete pinned OPUS-100 source-to-English test split with a FROZEN OPUS-MT model under a FIXED
beam-5 length-1.0 decode; the agent controls ONLY the SOURCE-side handling: the
maximum number of subword tokens the source is truncated to before encoding
(solution/tok.py -> build_source_max_tokens -> int). Scores corpus sacreBLEU / chrF.

Truncating the source too aggressively (e.g. 8-16 tokens) throws away the tail of
longer sentences -> untranslated content -> lower BLEU. A full-length window
(>= the corpus max, ~128) preserves the whole source and translates it all. This
is the "did you feed the model the whole sentence?" lever: over-short truncation
is a common, silent MT bug.

Emits:  MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
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

    srcs, refs = common.load_dataset()
    print(f"MT_DATA corpus=opus100_{common.direction()} n_pairs={len(srcs)}", flush=True)

    mx = common.require_int(
        common.load_surface_value(args.solution, "build_source_max_tokens"),
        "source_max_tokens",
        1,
        common.MAX_INPUT_TOKENS,
    )
    print(f"MT_TRUNC source_max_tokens={mx}", flush=True)

    gen_kwargs = {"num_beams": 5, "length_penalty": 1.0,
                  "early_stopping": True, "max_new_tokens": 128}
    model, tok = common.load_model_and_tokenizer(dev)
    preds = common.translate(model, tok, srcs, gen_kwargs, dev, max_input_tokens=mx)
    scores = common.score_bleu_chrf(preds, refs)
    plen = common.mean_pred_len_words(preds)
    dt = time.time() - t0
    print(f"MT_METRICS bleu={scores['bleu']:.6f} chrf={scores['chrf']:.6f} "
          f"n_pairs={len(srcs)} plen={plen:.1f} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
