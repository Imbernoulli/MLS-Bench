#!/usr/bin/env python3
"""One-off HOST-SIDE script: builds the frozen text-simplification test slices
(asset / turk / wiki, 300-sentence head-slice each -- the only slice any
shipped simp-* task ever requests) from the single ungated, no-remote-code
GEM/wiki_auto_asset_turk dataset, and splits each row into a SOURCE half and a
REFERENCES half before they ever touch an agent-visible or image-baked path.

Run this ONCE (or whenever the frozen regime below changes) from the repo
root, with HOST network access (the GEM dataset is pulled from the HF hub):

    python3 holdout/text-simplification/generate_data.py

Output split (class-3 fix, 2026-07-05): the source sentence is legitimate to
keep agent-visible (an agent must see the complex sentence to simplify it),
but the human reference simplifications are the literal answer key corpus
SARI is computed against, so they must NOT sit anywhere under the
agent-visible ``vendor/`` tree or get baked into the shared Harbor base image
via a pkg_config ``data_deps`` entry. This script writes two separate
locations per setting:

  * ``vendor/text-simplification/_simp_data/simp_<setting>_src.jsonl``
    -- ``{"source": "..."}`` rows ONLY (no ``references`` field at all).
    Stays in vendor/ (agent-visible at all times, checked into git); this is
    what every simp-* harness's ``common.py::load_dataset`` reads for
    sources via ``_SIMP_DATA_DIR``.
  * ``tasks/<task>/data/simp_<setting>_refs.jsonl`` (one copy per shipped
    simp-* task -- every task scores against all three settings, per each
    task's ``config.json`` ``test_cmds``) -- ``{"references": [...]}`` rows
    ONLY, same row order as the matching ``_src.jsonl``. Lives under each
    task's ``data/`` dir, which Harbor (and score_task.py's ``_task``
    symlink) stage into ``tests/meta/data`` and mount at
    ``/workspace/_task/data`` ONLY at verification time -- see
    ``harbor_adapter/src/mls_bench/adapter.py::_stage_verifier_assets`` and
    ``harbor_adapter/src/mls_bench/task-template/tests/test.sh``. The agent
    never sees this file during its action session; ``common.py::
    _refs_data_path`` + ``load_dataset`` SystemExit if it is absent.

This mirrors ``holdout/tpp-neural-hawkes/generate_data.py`` and
``holdout/ebm-langevin-cd/generate_data.py`` exactly, adapted to a
public-dataset re-serialisation split rather than a synthetic-DGP train/test
split (text-simplification has no generator secret to hide -- GEM/
wiki_auto_asset_turk is public and ungated -- only the held-out references
need gating).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # holdout/text-simplification
REPO_ROOT = HERE.parent.parent

VENDOR_DATA_DIR = REPO_ROOT / "vendor" / "text-simplification" / "_simp_data"
TASKS_DIR = REPO_ROOT / "tasks"

GEM_REPO = "GEM/wiki_auto_asset_turk"
SPLIT_MAP = {           # our setting name -> GEM split name
    "asset": "test_asset",
    "turk": "test_turk",
    "wiki": "test_wiki",
}

# Every shipped simp-* task scores against all three settings identically
# (each task's config.json test_cmds resolves asset/turk/wiki via one
# harness invocation), unlike mdn-density's per-task TASK_TARGETS subset --
# so a refs-only jsonl for every setting is written into every one of these
# tasks' data/ dirs (small files, cheap to duplicate; each task's data/ dir
# is staged/hidden independently by the adapter).
TASKS = (
    "simp-source-policy",
    "simp-decoding-beam",
    "simp-length-control",
    "simp-decoding-temperature",
    "simp-nucleus-sampling",
    "simp-decoding-strategy",
    "simp-input-truncation",
    "simp-model-capacity",
    "simp-beam-width",
    "simp-minlen-floor",
)


def _clean_refs(refs) -> list[str]:
    return [r.strip() for r in refs if r and r.strip()]


def _ok(src: str, refs: list[str]) -> bool:
    src = (src or "").strip()
    refs = _clean_refs(refs)
    if not src or not refs:
        return False
    if len(src.split()) < 3:            # too short to be a real sentence
        return False
    if len(src.split()) > 80:           # keep it minute-scale
        return False
    return True


def _write_refs_copy(task: str, setting: str, refs_rows: list[dict]) -> None:
    data_dir = TASKS_DIR / task / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"simp_{setting}_refs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in refs_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{setting}] wrote {out_path} (n={len(refs_rows)})")


def build_setting(setting: str, gem_split: str) -> None:
    from datasets import load_dataset

    ds = load_dataset(GEM_REPO, split=gem_split)
    src_rows = []
    refs_rows = []
    for r in ds:
        src = r["source"]
        refs = _clean_refs(list(r["references"]))
        if _ok(src, refs):
            src_rows.append({"source": src.strip()})
            refs_rows.append({"references": refs})

    VENDOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_path = VENDOR_DATA_DIR / f"simp_{setting}_src.jsonl"
    with src_path.open("w", encoding="utf-8") as f:
        for rec in src_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{setting}] wrote {src_path} (n={len(src_rows)})")

    for task in TASKS:
        _write_refs_copy(task, setting, refs_rows)

    avg_src = sum(len(x["source"].split()) for x in src_rows) / max(1, len(src_rows))
    avg_ref = sum(len(x["references"]) for x in refs_rows) / max(1, len(refs_rows))
    print(f"SIMP_BUILT setting={setting} split={gem_split} n={len(src_rows)} "
          f"avg_src_words={avg_src:.1f} avg_refs={avg_ref:.2f}", flush=True)


def main() -> None:
    for setting, gem_split in SPLIT_MAP.items():
        build_setting(setting, gem_split)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
