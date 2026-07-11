#!/usr/bin/env python3
"""summ-source-policy harness (fixed three-setting task).

For EACH of the THREE FIXED domain settings (xsum / cnndm / samsum), produces a
summary for every document by the agent's SOURCE POLICY (solution/source.py ->
build_source_policy -> a string), then scores corpus ROUGE-L F1 (and ROUGE-1/2 F1)
against fixed official references.

The policy chooses how each candidate summary is produced:
  "abstractive"   : decode the frozen domain-matched summarizer with a fixed
                    configuration.
  "lead3"         : use the first 1-3 source sentences.
  "copy_document" : use the beginning of the source document.
  "first_token"   : use only the first source word.
  "empty"         : return an empty string.

Emits one line per setting:
    SUMM_METRICS setting=<S> rougeL=<F> rouge1=<F> rouge2=<F> plen=<W>
"""
from __future__ import annotations

import argparse
import re
import time

import common

_VALID = {"abstractive", "lead3", "copy_document", "first_token", "empty"}


def _abstractive_gen(setting: str) -> dict:
    win = common.LEN_WINDOW[setting]
    beams = 6 if setting == "xsum" else 4
    return {
        "num_beams": beams,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        **win,
    }


def _lead_sentences(doc: str, k: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", doc.strip())
    return " ".join(parts[:k]) if parts else doc.strip()


def _truncate_words(doc: str, n: int = 120) -> str:
    return " ".join(doc.split()[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=common.SEED)
    args = ap.parse_args()

    dev = common.setup(args.seed)
    t0 = time.perf_counter()

    build_source_policy = common.load_surface(args.solution, "build_source_policy")
    policy = common.require_surface_choice(
        build_source_policy(), "policy", _VALID, surface="build_source_policy"
    )
    print(f"SUMM_SOURCE policy={policy}", flush=True)

    def preds_override(setting, docs):
        # Non-model policies produce predictions directly from the source text.
        if policy == "abstractive":
            return None  # let run_over_settings decode the model
        if policy == "lead3":
            k = 1 if setting == "xsum" else 3
            return [_lead_sentences(d, k) for d in docs]
        if policy == "copy_document":
            return [_truncate_words(d, 120) for d in docs]
        if policy == "first_token":
            return [(d.split()[0] if d.split() else "") for d in docs]
        return ["" for _ in docs]  # empty

    def build_gen(setting):
        return _abstractive_gen(setting)

    common.run_over_settings(build_gen, dev,
                             preds_override_for_setting=preds_override)
    print(
        f"SUMM_DONE settings={len(common.SETTINGS)} total_docs={common.TOTAL_DOCS} "
        f"seed={args.seed} elapsed={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
