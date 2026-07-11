#!/usr/bin/env python3
"""Prepare data + model for the constrained-decoding (cd-*) MLS-Bench tasks.

Produces, under {data_root}/constrained-decoding/:
  models/Qwen2.5-0.5B-Instruct/   FROZEN small instruction LM (all cd-* tasks)
  data/gsm8k.json                 FULL pinned GSM8K test split: [{question, gold}]
  data/classification.json        FIXED forced-choice set: {labels:[...], items:[{text,gold}]}

All fixtures are deterministic (seed 42). Requires network on the HOST; set
HF_ENDPOINT=https://hf-mirror.com if the default hub is unreachable.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

SEED = 42
N_GSM8K = 1319
N_CLS = 7600
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
AGNEWS_DATASET_ID = "fancyzhx/ag_news"
AGNEWS_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
SOURCE_MANIFEST = "source_manifest.json"

_GSM_ANS = re.compile(r"####\s*(-?[\d,]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _prepared_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted((root / "models" / "Qwen2.5-0.5B-Instruct").rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            hashes[path.relative_to(root).as_posix()] = _sha256(path)
    for name in ("gsm8k.json", "classification.json"):
        path = root / "data" / name
        hashes[path.relative_to(root).as_posix()] = _sha256(path)
    return hashes


def _manifest_metadata() -> dict:
    return {
        "schema_version": 1,
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gsm8k_repo": GSM8K_DATASET_ID,
        "gsm8k_revision": GSM8K_REVISION,
        "gsm8k_config": "main",
        "gsm8k_split": "test",
        "gsm8k_count": N_GSM8K,
        "agnews_repo": AGNEWS_DATASET_ID,
        "agnews_revision": AGNEWS_REVISION,
        "agnews_split": "test",
        "agnews_count": N_CLS,
        "agnews_selection_seed": SEED,
    }


def _prepared_is_valid(
    root: Path,
    *,
    expected_gsm8k: int = N_GSM8K,
    expected_classification: int = N_CLS,
) -> bool:
    try:
        manifest = json.loads((root / SOURCE_MANIFEST).read_text())
        metadata = _manifest_metadata()
        metadata["gsm8k_count"] = expected_gsm8k
        metadata["agnews_count"] = expected_classification
        if any(manifest.get(key) != value for key, value in metadata.items()):
            return False
        if manifest.get("prepared_sha256") != _prepared_hashes(root):
            return False
        gsm_sources = manifest.get("selected_sources", {}).get("gsm8k")
        cls_sources = manifest.get("selected_sources", {}).get("agnews")
        for rows, count in ((gsm_sources, expected_gsm8k),
                            (cls_sources, expected_classification)):
            if (not isinstance(rows, list) or len(rows) != count
                    or len({row.get("source_id") for row in rows
                            if isinstance(row, dict)}) != count
                    or any(not isinstance(row, dict)
                           or set(row) != {"source_id", "raw_sha256"}
                           or not isinstance(row["source_id"], str)
                           or not re.fullmatch(r"[0-9a-f]{64}", row["raw_sha256"])
                           for row in rows)):
                return False
        gsm8k = json.loads((root / "data" / "gsm8k.json").read_text())
        classification = json.loads((root / "data" / "classification.json").read_text())
        if (not isinstance(gsm8k, list) or len(gsm8k) != expected_gsm8k
                or any(not isinstance(row, dict) or set(row) != {"question", "gold"}
                       or not isinstance(row["question"], str) or not row["question"]
                       or not isinstance(row["gold"], str)
                       or not re.fullmatch(r"-?\d+", row["gold"])
                       for row in gsm8k)):
            return False
        if (not isinstance(classification, dict)
                or classification.get("labels") != ["World", "Sports", "Business", "Sci/Tech"]
                or not isinstance(classification.get("items"), list)
                or len(classification["items"]) != expected_classification
                or any(not isinstance(row, dict) or set(row) != {"text", "gold"}
                       or not isinstance(row["text"], str) or not row["text"]
                       or row["gold"] not in classification["labels"]
                       for row in classification["items"])):
            return False
        return (root / "models" / "Qwen2.5-0.5B-Instruct" / "config.json").is_file()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def download_model(models_dir: Path):
    from huggingface_hub import snapshot_download
    dst = models_dir / "Qwen2.5-0.5B-Instruct"
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(dst),
        ignore_patterns=["*.onnx", "*.ot", "*.h5", "*.msgpack", "onnx/*"],
    )
    print(f"downloaded {MODEL_ID}@{MODEL_REVISION} -> {dst}", flush=True)


def build_gsm8k(data_dir: Path):
    out = data_dir / "gsm8k.json"
    from datasets import load_dataset
    ds = load_dataset(
        GSM8K_DATASET_ID,
        "main",
        split="test",
        revision=GSM8K_REVISION,
    )
    items = []
    sources = []
    for source_index, row in enumerate(ds):
        m = _GSM_ANS.search(row["answer"])
        if not m:
            continue
        gold = m.group(1).replace(",", "")
        items.append({"question": row["question"].strip(), "gold": gold})
        sources.append({
            "source_id": f"test:{source_index}",
            "raw_sha256": _canonical_sha256({
                "question": row["question"], "answer": row["answer"]
            }),
        })
        if len(items) >= N_GSM8K:
            break
    if len(items) != N_GSM8K:
        raise RuntimeError(f"GSM8K selection incomplete: {len(items)}/{N_GSM8K}")
    out.write_text(json.dumps(items, ensure_ascii=False, indent=0))
    print(f"GSM8K_BUILT n={len(items)}", flush=True)
    return sources


def build_classification(data_dir: Path):
    out = data_dir / "classification.json"
    from datasets import load_dataset
    # AG News: full official 4-way test split. Labels are human-readable.
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    ds = load_dataset(
        AGNEWS_DATASET_ID,
        split="test",
        revision=AGNEWS_REVISION,
    )
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    items = []
    sources = []
    for i in idx:
        row = ds[i]
        text = row["text"].strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400]
        items.append({"text": text, "gold": id2label[row["label"]]})
        sources.append({
            "source_id": f"test:{i}",
            "raw_sha256": _canonical_sha256({
                "text": row["text"], "label": int(row["label"])
            }),
        })
        if len(items) >= N_CLS:
            break
    if len(items) != N_CLS:
        raise RuntimeError(f"AG News selection incomplete: {len(items)}/{N_CLS}")
    out.write_text(json.dumps({"labels": labels, "items": items},
                              ensure_ascii=False, indent=0))
    # quick majority-class report
    from collections import Counter
    c = Counter(it["gold"] for it in items)
    print(f"CLS_BUILT n={len(items)} dist={dict(c)} "
          f"majority={max(c.values())/len(items):.3f}", flush=True)
    return sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    args = ap.parse_args()

    configured_root = Path(args.data_root) / "constrained-decoding"
    # The adapter data registry may expose package data through a symlink. Keep
    # that registry link intact and atomically replace its real target instead.
    root = configured_root.resolve() if configured_root.is_symlink() else configured_root
    if _prepared_is_valid(root):
        print(f"constrained-decoding data already present and fully validated at {root}")
        return 0
    staging = root.with_name(f"{root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    models_dir = staging / "models"
    data_dir = staging / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    download_model(models_dir)
    gsm_sources = build_gsm8k(data_dir)
    cls_sources = build_classification(data_dir)
    manifest = _manifest_metadata()
    manifest["selected_sources"] = {"gsm8k": gsm_sources, "agnews": cls_sources}
    manifest["prepared_sha256"] = _prepared_hashes(staging)
    (staging / SOURCE_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    if not _prepared_is_valid(staging):
        raise RuntimeError("prepared constrained-decoding assets failed validation")
    if root.exists():
        shutil.rmtree(root)
    staging.replace(root)
    print(f"constrained-decoding data ready at {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
