#!/usr/bin/env python3
"""Prepare the FROZEN model for the text-simplification (simp-*) MLS-Bench tasks.

Produces, under {data_root}/text-simplification/:
  models/t5-base-finetuned-turk-text-simplification/  FROZEN pretrained T5 simplifier

The simplifier is ``mrm8488/t5-base-finetuned-turk-text-simplification`` (a T5-base
fine-tuned on the Wiki/Turk simplification data; expects a ``simplify: `` prefix and
emits a simplified sentence, ~220M params), pulled from the HF hub.

## Held-out-reference hygiene (class-3 fix, 2026-07-05, PR #54 pattern)

This script is a pkg_config ``data_deps[].prepare`` entry, i.e. its OUTPUT
directory (``{data_root}/text-simplification``) is exactly what
``harbor_adapter/scripts/build_base_image.py`` bakes into the SHARED base
image every simp-* task's agent shell can read. It therefore now stages the
MODEL ONLY -- no test-set JSONL of any kind. The three FIXED simplification
test sets (source sentences + held-out human reference simplifications, from
GEM/wiki_auto_asset_turk) are built separately by
``holdout/text-simplification/generate_data.py``, which writes the
sources-only half to the agent-visible, git-checked-in
``vendor/text-simplification/_simp_data/`` and the references-only half
(the literal SARI answer key) DIRECTLY into each shipped ``tasks/simp-*/
data/`` dir (verifier-only, mounted at ``$TASK_DIR/data`` only at scoring
time). See ``vendor/text-simplification/common.py::load_dataset`` /
``_refs_data_path`` for the loader side of this split.

Requires network on the HOST; the task container is offline.
"""
import argparse
from pathlib import Path

MODEL_REPO = "mrm8488/t5-base-finetuned-turk-text-simplification"


def download_model(models_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    dst = models_dir / "t5-base-finetuned-turk-text-simplification"
    if dst.exists() and (dst / "config.json").exists():
        print(f"model {MODEL_REPO} already present", flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(dst),
        ignore_patterns=["*.onnx", "*.ot", "*.h5", "*.msgpack", "tf_model.h5",
                         "rust_model.ot", "onnx/*", "openvino/*", "*.tflite",
                         "flax_model.msgpack", "runs/*", "*.tfevents.*"],
    )
    print(f"  downloaded {MODEL_REPO} -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "text-simplification"
    (root / "models").mkdir(parents=True, exist_ok=True)
    download_model(root / "models")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
