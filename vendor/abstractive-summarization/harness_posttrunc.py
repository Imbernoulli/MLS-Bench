#!/usr/bin/env python3
"""summ-post-truncation harness (3-setting).

Post-processing sentence selection is applied after one fixed model decode
+ per-domain length window) is FIXED and IDENTICAL for every config; the agent
chooses only how many leading SENTENCES of the decoded summary to KEEP
(solution/posttrunc.py -> build_keep_sentences -> int; a large value keeps all).

This is a pure post-process: the model output is identical across configurations,
and only the retained sentence prefix differs. The three domains have different
reference sentence distributions, so the benchmark measures the shared policy
without publishing a preferred retention count. Uses mean per-example ROUGE-L F1.

Emits one line per setting:
    SUMM_METRICS setting=<S> rougeL=<F> rouge1=<F> rouge2=<F> plen=<W>
"""
from __future__ import annotations

import argparse
import re
import time

import common


def _keep_first_sentences(text: str, k: int) -> str:
    if k <= 0:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:k]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_keep_sentences = common.load_surface(args.solution, "build_keep_sentences")
    keep = common.require_surface_int(
        build_keep_sentences(), "keep_sentences", 0, 10000,
        surface="build_keep_sentences",
    )
    print(f"SUMM_POSTTRUNC keep_sentences={keep}", flush=True)
    print(
        f"SUMM_PROTOCOL version={common.PROTOCOL} "
        f"settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS}",
        flush=True,
    )

    for setting in common.SETTINGS:
        docs, refs = common.load_dataset(setting)
        model, tok = common.load_model_and_tokenizer(setting, dev)
        win = common.LEN_WINDOW[setting]
        beams = 6 if setting == "xsum" else 4
        gk = {"num_beams": beams, "no_repeat_ngram_size": 3,
              "early_stopping": True, **win}
        raw = common.generate_summaries(
            model, tok, docs, gk, dev, setting=setting
        )
        if keep >= 999:
            preds = raw
        else:
            preds = [_keep_first_sentences(p, keep) for p in raw]
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores = common.score_rouge(preds, refs)
        plen = common.mean_pred_len_words(preds)
        common.emit_metrics(setting, scores, plen, len(docs))
        print(
            f"SUMM_SETTING_DONE setting={setting} generated={len(preds)} "
            f"expected={common.DATASET_INVENTORY[setting]['rows']}",
            flush=True,
        )

    print(
        f"SUMM_EVAL_DONE settings={len(common.SETTINGS)} "
        f"total_docs={common.TOTAL_DOCS}",
        flush=True,
    )
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
