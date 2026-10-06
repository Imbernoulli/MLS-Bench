#!/usr/bin/env python3
"""Download Mistral-7B-v0.1 weights from HuggingFace for PTQ tasks.

Downloads the ``mistralai/Mistral-7B-v0.1`` model (~14GB) so that compute
nodes (which have no network access) can load pretrained weights directly.

Output: {data_root}/mistral-7b-v01/  (HuggingFace snapshot)

Usage:
    python vendor/data_scripts/gptq/prepare_model.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data gptq
"""

import argparse
import os
import sys
from pathlib import Path


REPO_ID = "mistralai/Mistral-7B-v0.1"


def main():
    parser = argparse.ArgumentParser(description="Download Mistral-7B-v0.1 weights")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "mistral-7b-v01"
    dest.mkdir(parents=True, exist_ok=True)

    if dest.exists() and any(dest.iterdir()):
        print("mistral-7b-v01: already downloaded, skipping")
        return

    print(f"Downloading {REPO_ID}...", flush=True)
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(dest))
        print("mistral-7b-v01: done")
    except Exception as e:
        print(f"mistral-7b-v01: FAILED - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nMistral-7B-v0.1 weights ready at {dest}/")


if __name__ == "__main__":
    main()
