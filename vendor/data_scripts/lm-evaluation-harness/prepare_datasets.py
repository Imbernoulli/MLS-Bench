#!/usr/bin/env python3
"""Prepare HuggingFace datasets for lm-evaluation-harness offline evaluation.

Downloads datasets used by lm-eval benchmarks (hellaswag, arc_easy, piqa,
winogrande) under all known name variants so offline cache lookup succeeds
regardless of HF Hub redirects.

Output: {data_root}/lm-eval-datasets/  (~200 MB)

Usage:
    python vendor/data_scripts/lm-evaluation-harness/prepare_datasets.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data lm-evaluation-harness
"""

import argparse
import sys
import time
from pathlib import Path


DATASETS = [
    ("hellaswag", None),
    ("Rowan/hellaswag", None),
    ("piqa", None),
    ("baber/piqa", None),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    # lm-eval renamed winogrande's dataset_path from the bare `winogrande`
    # (old convention) to the namespaced `allenai/winogrande` (>=0.4.x).
    # Cache both names so offline lookup matches regardless of which lm-eval
    # version the eval image ends up resolving — same pattern as hellaswag /
    # piqa above. Without the namespaced variant, an image built with a newer
    # lm-eval looks for `allenai___winogrande/` and fails offline.
    ("winogrande", "winogrande_xl"),
    ("allenai/winogrande", "winogrande_xl"),
]


def load_with_retry(load_dataset, name, config, cache_dir, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(
                name,
                config,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
            )
        except Exception:
            if attempt == attempts:
                raise
            print(
                f"  Retry {attempt}/{attempts - 1}: {name} after transient failure",
                file=sys.stderr,
            )
            time.sleep(5 * attempt)


def main():
    parser = argparse.ArgumentParser(description="Download lm-eval datasets")
    parser.add_argument(
        "--data-root", type=str, default="vendor/data",
        help="Root directory for data storage",
    )
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "lm-eval-datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading datasets to: {cache_dir}")

    from datasets import load_dataset

    for name, config in DATASETS:
        try:
            print(f"  Loading {name} (config={config})...")
            load_with_retry(load_dataset, name, config, cache_dir)
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  FAIL: {name}: {e}", file=sys.stderr)
            sys.exit(1)

    (cache_dir / ".complete").write_text(
        "\n".join(
            f"{name}:{config or 'default'}" for name, config in DATASETS
        ) + "\n"
    )
    print(f"\nAll datasets downloaded to {cache_dir}")


if __name__ == "__main__":
    main()
