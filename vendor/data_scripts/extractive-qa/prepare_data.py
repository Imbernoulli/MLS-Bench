#!/usr/bin/env python3
"""Stage the complete official validation data and frozen QA checkpoint.

The generated JSONL files are deterministic re-serializations of the complete
MRQA validation subsets and the complete SQuAD v2 validation split.  SQuAD v2
is partitioned by original row index modulo three so that every official row is
evaluated exactly once across the three configured settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download


MRQA_REPO = "mrqa-workshop/mrqa"
MRQA_REVISION = "f3178d9888471dfb2b67c93de14f0ddf499a8d9f"
MRQA_PARQUET = "plain_text/validation-00000-of-00001.parquet"

SQUAD2_REPO = "rajpurkar/squad_v2"
SQUAD2_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"
SQUAD2_PARQUET = "squad_v2/validation-00000-of-00001.parquet"

MODEL_REPO = "deepset/roberta-base-squad2"
MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
MODEL_FILES = {
    "config.json": "64fa58495a722d57609c22f199824bfe98c19be068136a70c268214a08cb8060",
    "model.safetensors": "ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
    "tokenizer_config.json": "7a33226d4265e3989cc6341666af179d0cc710136f4059aae0dd8c0797cba556",
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
}

MRQA_DOMAINS = {
    "SQuAD": (
        "mrqa_squad_validation.jsonl",
        10_507,
        "64ab3a4c69574a258c934044a63605b15d98e1608fa9fb5b244868c5d0af89aa",
    ),
    "NewsQA": (
        "mrqa_newsqa_validation.jsonl",
        4_212,
        "87b31cff3db4cb8276ddc58c94b03ca3ca500a72af95b8b9e2c63c9266ded7ad",
    ),
    "HotpotQA": (
        "mrqa_hotpotqa_validation.jsonl",
        5_901,
        "a335e1778d3c2de3a99b00e8eeaa3fc6e9b611386afadcc54532c2f33d3d95ad",
    ),
    "NaturalQuestionsShort": (
        "mrqa_naturalquestions_validation.jsonl",
        12_836,
        "705717e225fc972d9a1df01737ab11d59a2c573a6ba9e7018b5ace4c34de6952",
    ),
}

SQUAD2_PARTS = {
    0: (
        "squad2_validation_part0.jsonl",
        3_958,
        1_988,
        1_970,
        "bdb7f256bf8893edef347623c6698a16320608d5ddf31c774de8e8234598f5b9",
    ),
    1: (
        "squad2_validation_part1.jsonl",
        3_958,
        1_956,
        2_002,
        "4159c7c652415873aa565af317a8c0d460164b5f80b185a35b9cbe6dac40f327",
    ),
    2: (
        "squad2_validation_part2.jsonl",
        3_957,
        1_984,
        1_973,
        "4b8fff6cb1dd3370416e1cf36cb7d8ba846ef61fd2cb086ccd02ad80a97ce651",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, int, int]:
    count = answerable = unanswerable = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            if row["is_impossible"]:
                unanswerable += 1
            else:
                answerable += 1
    return count, answerable, unanswerable


def _require_fixture(
    path: Path,
    *,
    count: int,
    answerable: int,
    unanswerable: int,
    sha256: str,
    observed: tuple[int, int, int] | None = None,
) -> None:
    if observed is None:
        observed_count = observed_answerable = observed_unanswerable = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                observed_count += 1
                if row["is_impossible"]:
                    observed_unanswerable += 1
                else:
                    observed_answerable += 1
        observed = (observed_count, observed_answerable, observed_unanswerable)
    expected = (count, answerable, unanswerable)
    if observed != expected or _sha256(path) != sha256:
        raise RuntimeError(
            f"generated fixture mismatch for {path.name}: "
            f"counts={observed} expected={expected} sha256={_sha256(path)}"
        )


def download_model(models_dir: Path) -> None:
    destination = models_dir / "roberta-base-squad2"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(destination),
        allow_patterns=sorted(MODEL_FILES),
    )
    for filename, expected_sha in MODEL_FILES.items():
        path = destination / filename
        if not path.is_file() or _sha256(path) != expected_sha:
            raise RuntimeError(f"model file mismatch: {filename}")
    print(f"MODEL_READY revision={MODEL_REVISION}", flush=True)


def build_mrqa(data_dir: Path) -> None:
    import pandas as pd

    parquet = hf_hub_download(
        repo_id=MRQA_REPO,
        repo_type="dataset",
        revision=MRQA_REVISION,
        filename=MRQA_PARQUET,
    )
    frame = pd.read_parquet(parquet)
    for subset, (filename, count, expected_sha) in MRQA_DOMAINS.items():
        path = data_dir / filename

        def rows():
            for _, record in frame[frame["subset"] == subset].iterrows():
                answers = _deduplicate(record["answers"])
                if not answers:
                    raise RuntimeError(f"MRQA answer missing for {record['qid']}")
                yield {
                    "id": str(record["qid"]),
                    "question": str(record["question"]).strip(),
                    "context": str(record["context"]),
                    "answers": answers,
                    "is_impossible": False,
                }

        observed = _write_jsonl(path, rows())
        _require_fixture(
            path,
            count=count,
            answerable=count,
            unanswerable=0,
            sha256=expected_sha,
            observed=observed,
        )
        print(f"DATA_READY file={filename} n={count} sha256={expected_sha}", flush=True)


def build_squad2(data_dir: Path) -> None:
    import pandas as pd

    parquet = hf_hub_download(
        repo_id=SQUAD2_REPO,
        repo_type="dataset",
        revision=SQUAD2_REVISION,
        filename=SQUAD2_PARQUET,
    )
    frame = pd.read_parquet(parquet)
    all_ids: list[str] = []
    for part, (filename, count, n_ans, n_noans, expected_sha) in SQUAD2_PARTS.items():
        path = data_dir / filename

        def rows():
            for index, (_, record) in enumerate(frame.iterrows()):
                if index % 3 != part:
                    continue
                raw_answers = record["answers"]
                answers = (
                    _deduplicate(raw_answers["text"])
                    if isinstance(raw_answers, dict)
                    else []
                )
                qid = str(record["id"])
                all_ids.append(qid)
                yield {
                    "id": qid,
                    "question": str(record["question"]).strip(),
                    "context": str(record["context"]),
                    "answers": answers,
                    "is_impossible": not answers,
                }

        observed = _write_jsonl(path, rows())
        _require_fixture(
            path,
            count=count,
            answerable=n_ans,
            unanswerable=n_noans,
            sha256=expected_sha,
            observed=observed,
        )
        print(f"DATA_READY file={filename} n={count} sha256={expected_sha}", flush=True)

    official_ids = [str(value) for value in frame["id"].tolist()]
    if len(all_ids) != len(official_ids) or set(all_ids) != set(official_ids):
        raise RuntimeError("SQuAD v2 partition does not exactly cover validation")
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("SQuAD v2 validation contains duplicate IDs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(args.data_root) / "extractive-qa"
    models_dir = root / "models"
    data_dir = root / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    download_model(models_dir)
    build_mrqa(data_dir)
    build_squad2(data_dir)
    print("ALL_DONE protocol=qa-official-full-v2", flush=True)


if __name__ == "__main__":
    main()
