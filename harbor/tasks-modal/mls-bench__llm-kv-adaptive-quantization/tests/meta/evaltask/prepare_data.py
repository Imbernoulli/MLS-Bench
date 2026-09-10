#!/usr/bin/env python3
"""Prepare offline Hugging Face assets for adaptive KV quantization."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LONGBENCH_REPO = "THUDM/LongBench"
NEEDLEBENCH_REPO = "opencompass/NeedleBench"


def configure_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)


def download_model() -> None:
    from huggingface_hub import snapshot_download

    print(f"[adaptive-quant] downloading model: {MODEL_ID}", flush=True)
    snapshot_download(repo_id=MODEL_ID)


def download_benchmark_files() -> None:
    from huggingface_hub import hf_hub_download

    print(f"[adaptive-quant] downloading {LONGBENCH_REPO}/data.zip", flush=True)
    hf_hub_download(
        repo_id=LONGBENCH_REPO,
        filename="data.zip",
        repo_type="dataset",
    )
    print(f"[adaptive-quant] downloading {NEEDLEBENCH_REPO}/PaulGrahamEssays.jsonl", flush=True)
    hf_hub_download(
        repo_id=NEEDLEBENCH_REPO,
        filename="PaulGrahamEssays.jsonl",
        repo_type="dataset",
    )


def download_gsm8k() -> None:
    from datasets import load_dataset

    print("[adaptive-quant] downloading openai/gsm8k main test split", flush=True)
    load_dataset("openai/gsm8k", "main", split="test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare adaptive KV quantization data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("vendor/data"),
        help="MLS-Bench data root",
    )
    args = parser.parse_args()

    cache_root = args.data_root / "huggingface_cache"
    configure_cache(cache_root)

    try:
        download_model()
        download_benchmark_files()
        download_gsm8k()
    except Exception as exc:
        print(f"[adaptive-quant] data preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[adaptive-quant] offline cache ready: {cache_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
