#!/usr/bin/env python3
"""Prepare data + models for the machine-translation (mt-*) MLS-Bench tasks.

Produces, under {data_root}/machine-translation/:
  models/opus-mt-de-en/    FROZEN small pretrained MarianMT (German  -> English)
  models/opus-mt-fr-en/    FROZEN small pretrained MarianMT (French  -> English)
  models/opus-mt-ru-en/    FROZEN small pretrained MarianMT (Russian -> English)
  data/de_en_test.jsonl    complete official 2000-pair de->en test split
  data/fr_en_test.jsonl    complete official 2000-pair fr->en test split
  data/ru_en_test.jsonl    complete official 2000-pair ru->en test split

The MT models are the Helsinki-NLP OPUS-MT MarianMT models (Tiedemann &
Thottingal 2020), ~75M params each, pulled from the HF hub. The test sets are
the complete official OPUS-100 <src>-en test splits (Zhang et al. 2020,
"Improving Massively Multilingual NMT"; the ungated, script-free HF parquet
dataset ``Helsinki-NLP/opus-100``, split ``test``, 2000 pairs each). We preserve
every official row in its original order and serialise the aligned pairs to JSONL

    {"src": "<source sentence>", "ref": "<English reference>"}

so the task container resolves the corpus fully offline (no network at task time).
Requires network on the HOST; the task container is offline.

All three directions translate INTO English so the sacreBLEU reference side is a
single language; the mt-* tasks aggregate their metric (geometric mean) over the
three directions (de_en / fr_en / ru_en) as their >=3 validation settings.

(OPUS-100 is used rather than FLORES because it is ungated, script-free parquet,
and from the same OPUS family as the frozen models.)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DATASET_REVISION = "805090dc28bf78897da9641cdf08b61287580df9"
EXPECTED_PAIRS = 2000
MODEL_REVISIONS = {
    "Helsinki-NLP/opus-mt-de-en": "1a922f3b32a8e809e17a47d4b32142d8105924e5",
    "Helsinki-NLP/opus-mt-fr-en": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
    "Helsinki-NLP/opus-mt-ru-en": "fbd6dc73284f95536648512cc21d57f19191961a",
}
EXPECTED_SHA256 = {
    "de_en_test.jsonl": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
    "fr_en_test.jsonl": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
    "ru_en_test.jsonl": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
}

# (direction key, model repo, opus-100 config, source lang code, model dir, out file)
DIRECTIONS = [
    ("de_en", "Helsinki-NLP/opus-mt-de-en", "de-en", "de", "opus-mt-de-en", "de_en_test.jsonl"),
    ("fr_en", "Helsinki-NLP/opus-mt-fr-en", "en-fr", "fr", "opus-mt-fr-en", "fr_en_test.jsonl"),
    ("ru_en", "Helsinki-NLP/opus-mt-ru-en", "en-ru", "ru", "opus-mt-ru-en", "ru_en_test.jsonl"),
]


def download_model(models_dir: Path, repo: str, model_dir: str) -> None:
    from huggingface_hub import snapshot_download

    dst = models_dir / model_dir
    if dst.exists() and (dst / "config.json").exists():
        print(f"model {repo} already present", flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        revision=MODEL_REVISIONS[repo],
        local_dir=str(dst),
        ignore_patterns=["*.onnx", "*.ot", "*.h5", "*.msgpack", "tf_model.h5",
                         "rust_model.ot", "onnx/*", "openvino/*", "*.tflite",
                         "flax_model.msgpack"],
    )
    print(f"  downloaded {repo} -> {dst}", flush=True)


def _load_src_en(cfg: str, src_lang: str):
    """Return (src_sents, en_sents) aligned lists from OPUS-100 <cfg> test."""
    from datasets import load_dataset

    ds = load_dataset(
        "Helsinki-NLP/opus-100",
        cfg,
        split="test",
        revision=DATASET_REVISION,
    )
    src_sents, en_sents = [], []
    for r in ds:
        t = r["translation"]
        src_sents.append(t[src_lang])
        en_sents.append(t["en"])
    print(f"  loaded OPUS-100 {cfg} test ({len(src_sents)} raw pairs)", flush=True)
    return src_sents, en_sents


def _has_expected_file(path: Path, expected_sha256: str | None = None) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 is not None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            return False
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return len(rows) == EXPECTED_PAIRS and all(
        isinstance(row, dict)
        and set(row) == {"src", "ref"}
        and isinstance(row["src"], str)
        and bool(row["src"].strip())
        and isinstance(row["ref"], str)
        and bool(row["ref"].strip())
        for row in rows
    )


def build_testset(
    data_dir: Path,
    cfg: str,
    src_lang: str,
    out_name: str,
    expected_sha256: str | None = None,
) -> None:
    out = data_dir / out_name
    if _has_expected_file(out, expected_sha256):
        print(f"complete official {out_name} already present", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    src_sents, en_sents = _load_src_en(cfg, src_lang)
    if len(src_sents) != EXPECTED_PAIRS or len(en_sents) != EXPECTED_PAIRS:
        raise ValueError(
            f"pinned OPUS-100 {cfg} test split must contain {EXPECTED_PAIRS} pairs, "
            f"got {len(src_sents)} sources and {len(en_sents)} references"
        )
    pairs = []
    for index, (source, reference) in enumerate(zip(src_sents, en_sents), 1):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"invalid source in {cfg} test row {index}")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"invalid reference in {cfg} test row {index}")
        pairs.append({"src": source.strip(), "ref": reference.strip()})

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    tmp.replace(out)
    if expected_sha256 is not None:
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"prepared {out_name} digest mismatch: expected {expected_sha256}, got {digest}"
            )

    avg_src = sum(len(r["src"].split()) for r in pairs) / len(pairs)
    avg_ref = sum(len(r["ref"].split()) for r in pairs) / len(pairs)
    print(f"TESTSET_BUILT {out_name} pairs={len(pairs)} avg_src_words={avg_src:.1f} "
          f"avg_ref_words={avg_ref:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(args.data_root) / "machine-translation"
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    for _key, repo, cfg, src_lang, model_dir, out_name in DIRECTIONS:
        download_model(root / "models", repo, model_dir)
        build_testset(
            root / "data",
            cfg,
            src_lang,
            out_name,
            EXPECTED_SHA256[out_name],
        )

    manifest = {
        "format": "mls-bench-opus100-official-test-v1",
        "dataset": "Helsinki-NLP/opus-100",
        "dataset_revision": DATASET_REVISION,
        "pairs_per_direction": EXPECTED_PAIRS,
        "directions": [key for key, *_rest in DIRECTIONS],
        "model_revisions": MODEL_REVISIONS,
        "sha256": EXPECTED_SHA256,
    }
    (root / "data" / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
