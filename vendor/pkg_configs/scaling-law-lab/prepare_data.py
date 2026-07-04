#!/usr/bin/env python3
"""Download the SLDBench subsets used by the task at build time.

The TRAIN splits are written in full. The TEST splits are written with their
target column(s) STRIPPED: the in-container program only ever needs the test
FEATURES to produce predictions, and the held-out targets are joined outside
the agent's process (natively by the task parser from holdout/, in Harbor by
the verifier from tests/meta/). Shipping test targets into the container let
an edited model read the answers straight off disk.

The dataset revision is pinned so the baked features, the committed held-out
targets, and any future rebuild all describe the same rows.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset


# Container default is /data/scaling_law; local mode overrides via env.
ROOT = Path(os.environ.get("SCALING_LAW_DATA_DIR", "/data/scaling_law"))
ROOT.mkdir(parents=True, exist_ok=True)

# Pin the dataset snapshot (drift guard: an unpinned re-download could
# silently change rows and desynchronize features from the held-out targets).
REVISION = "721b846056f031737ff7fa72572c021324e3ec0e"

# Per-config target column(s), including the generic fallback key the
# benchmark loader also accepts. Stripped from the TEST split only.
TARGET_KEYS = {
    "vocab_scaling_law": ("unigram_normalized_loss", "loss"),
    "lr_bsz_scaling_law": ("lm_loss", "loss"),
    "data_constrained_scaling_law": ("loss",),
}


def dump_dataset(dataset_name: str, config_name: str, split: str, prefix: str) -> int:
    ds = load_dataset(dataset_name, config_name, split=split, revision=REVISION)
    path = ROOT / f"{prefix}__{config_name}__{split}.jsonl"
    if split == "test":
        drop = set(TARGET_KEYS[config_name])
        with path.open("w") as f:
            for row in ds:
                f.write(json.dumps({k: v for k, v in row.items() if k not in drop}) + "\n")
        print(
            f"Saved {dataset_name}/{config_name}/{split} -> {path} "
            f"({len(ds)} rows, targets stripped: {sorted(drop)})",
            flush=True,
        )
    else:
        ds.to_json(str(path))
        print(f"Saved {dataset_name}/{config_name}/{split} -> {path} ({len(ds)} rows)", flush=True)
    return len(ds)


# Harder SLDBench subsets recommended by sldbench authors (see SLDAgent paper).
# NOTE: `lr_bsz_scaling_law_modified` is declared in the HF README but its
# parquet files are missing upstream (404 on resolve/main). We fall back to the
# published `lr_bsz_scaling_law` subset, which has the same schema. Revisit if
# upstream publishes the modified variant.
manifest = {}
for cfg in (
    "vocab_scaling_law",
    "lr_bsz_scaling_law",
    "data_constrained_scaling_law",
):
    for split in ("train", "test"):
        manifest[f"{cfg}/{split}"] = dump_dataset("pkuHaowei/sldbench", cfg, split, "sldbench")

(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
