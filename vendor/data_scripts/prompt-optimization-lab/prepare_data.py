"""Prepare the frozen model and canonical full-scale APE evaluation data.

All expensive work performed by this module belongs in an image-build worker. The
result is copied into the per-repository image; verification runs fully offline.

Protocol:
  * Qwen/Qwen2.5-0.5B-Instruct, frozen and inference-only.
  * AG News: 128 proposal + 200 selection rows from the official train split,
    then every one of the 7,600 official test rows for the scored evaluation.
  * SST-2: 128 proposal + 200 selection rows from the official train split,
    then every one of the 872 official validation rows for scored evaluation.
  * Proposal, selection, and evaluation texts are pairwise disjoint. Candidate
    ranking never reads evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
import urllib.request
from pathlib import Path


PROTOCOL = "ape_qwen25_05b_full_official_v1"
SEED = 42
N_POOL = 128
N_DEV = 200
LM_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_EVAL = {"agnews": 7_600, "sst2": 872}
AG_NEWS_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
GLUE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha_bytes(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _snapshot(
    repo_id: str,
    dst: Path,
    model_src: str | None = None,
    *,
    reference_model: bool = False,
) -> None:
    if dst.exists() and (dst / "config.json").exists():
        print(f"model already present -> {dst}", flush=True)
        return
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    if model_src:
        src = Path(model_src)
        if not (src / "config.json").is_file():
            raise SystemExit(f"--model-src {src} has no config.json")
        if reference_model:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src, target_is_directory=True)
            print(f"referenced frozen model {src} -> {dst}", flush=True)
            return
        shutil.copytree(src, dst, symlinks=False)
        print(f"copied frozen model {src} -> {dst}", flush=True)
        return

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dst),
        ignore_patterns=[
            "*.onnx",
            "*.ot",
            "*.h5",
            "*.msgpack",
            "*.gguf",
            "openvino/*",
            "onnx/*",
        ],
    )
    print(f"downloaded {repo_id} -> {dst}", flush=True)


def _unique_rows(rows, text_key: str, blocked_texts: set[str]) -> list[dict]:
    unique: list[dict] = []
    seen = set(blocked_texts)
    for row in rows:
        text = _normalize(row[text_key])
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append({"text": text, "label": int(row["label"])})
    return unique


def _balanced_take(
    rows: list[dict], n: int, rng: random.Random, used_texts: set[str]
) -> list[dict]:
    labels = sorted({int(row["label"]) for row in rows})
    if n % len(labels):
        raise ValueError(f"balanced inventory {n} is not divisible by {len(labels)}")
    per_label = n // len(labels)
    selected: list[dict] = []
    for label in labels:
        candidates = [
            row for row in rows
            if int(row["label"]) == label and row["text"] not in used_texts
        ]
        rng.shuffle(candidates)
        if len(candidates) < per_label:
            raise ValueError(f"not enough unique train rows for label {label}")
        chosen = candidates[:per_label]
        selected.extend(chosen)
        used_texts.update(row["text"] for row in chosen)
    rng.shuffle(selected)
    return selected


def _write_rows(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(rows)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"wrote {len(rows)} rows sha256={digest} -> {path}", flush=True)
    return digest


def _write_dataset(
    *,
    data_root: Path,
    name: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    eval_split: str,
    labels: dict[int, str],
    task: str,
    source: str,
) -> dict:
    expected_eval = EXPECTED_EVAL[name]
    if len(eval_rows) != expected_eval:
        raise ValueError(
            f"{name} official {eval_split} inventory is {len(eval_rows)}, "
            f"expected {expected_eval}"
        )
    eval_texts = {row["text"] for row in eval_rows}

    rng = random.Random(SEED)
    used = set(eval_texts)
    pool = _balanced_take(train_rows, N_POOL, rng, used)
    dev = _balanced_take(train_rows, N_DEV, rng, used)
    pool_texts = {row["text"] for row in pool}
    dev_texts = {row["text"] for row in dev}
    if pool_texts & dev_texts or pool_texts & eval_texts or dev_texts & eval_texts:
        raise ValueError(f"{name} proposal/selection/evaluation splits are not disjoint")

    out = data_root / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {
        "pool.json": _write_rows(pool, out / "pool.json"),
        "dev.json": _write_rows(dev, out / "dev.json"),
        "test.json": _write_rows(eval_rows, out / "test.json"),
    }
    meta = {
        "dataset": name,
        "source": source,
        "seed": SEED,
        "labels": labels,
        "n_class": len(labels),
        "task": task,
        "proposal_split": "train",
        "selection_split": "train",
        "evaluation_split": eval_split,
        "pool_n": N_POOL,
        "dev_n": N_DEV,
        "eval_n": expected_eval,
        "pairwise_text_disjoint": True,
    }
    meta_payload = _canonical_bytes(meta)
    (out / "meta.json").write_bytes(meta_payload)
    files["meta.json"] = hashlib.sha256(meta_payload).hexdigest()
    record = {**meta, "files": files}
    record["data_sha256"] = _sha_bytes(record)
    return record


def _existing_dataset_record(data_root: Path, name: str) -> dict | None:
    out = data_root / name
    required = [out / filename for filename in ("pool.json", "dev.json", "test.json", "meta.json")]
    if not all(path.is_file() for path in required):
        return None
    try:
        meta = json.loads((out / "meta.json").read_text())
        counts = {
            "pool_n": len(json.loads((out / "pool.json").read_text())),
            "dev_n": len(json.loads((out / "dev.json").read_text())),
            "eval_n": len(json.loads((out / "test.json").read_text())),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        counts != {"pool_n": N_POOL, "dev_n": N_DEV, "eval_n": EXPECTED_EVAL[name]}
        or any(meta.get(key) != value for key, value in counts.items())
        or meta.get("pairwise_text_disjoint") is not True
    ):
        return None
    expected_source = (
        f"ag_news@{AG_NEWS_REVISION}"
        if name == "agnews"
        else f"glue/sst2@{GLUE_REVISION}"
    )
    if meta.get("source") != expected_source:
        meta["source"] = expected_source
        (out / "meta.json").write_bytes(_canonical_bytes(meta))
    files = {path.name: _sha_file(path) for path in required}
    record = {**meta, "files": files}
    record["data_sha256"] = _sha_bytes(record)
    print(f"reused complete {name} inventory from {out}", flush=True)
    return record


def _download_parquet(root: Path, repo: str, revision: str, filename: str) -> Path:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    dst = root / "_downloads" / repo.replace("/", "__") / revision / filename
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{filename}"
    error = None
    for attempt in range(1, 6):
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mlsbench-promptopt/1"})
            with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            if tmp.stat().st_size <= 0:
                raise OSError("downloaded parquet is empty")
            os.replace(tmp, dst)
            print(f"downloaded {url} bytes={dst.stat().st_size}", flush=True)
            return dst
        except Exception as exc:
            error = exc
            tmp.unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to download pinned parquet {url}: {error}")


def _parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def build_data(data_root: Path) -> dict:
    ag_record = _existing_dataset_record(data_root, "agnews")
    if ag_record is None:
        ag_train_path = _download_parquet(
            data_root, "ag_news", AG_NEWS_REVISION, "data/train-00000-of-00001.parquet"
        )
        ag_test_path = _download_parquet(
            data_root, "ag_news", AG_NEWS_REVISION, "data/test-00000-of-00001.parquet"
        )
        ag_train_raw = _parquet_rows(ag_train_path)
        ag_eval = [
            {"text": _normalize(row["text"]), "label": int(row["label"])}
            for row in _parquet_rows(ag_test_path)
        ]
        ag_train = _unique_rows(
            ag_train_raw, "text", {row["text"] for row in ag_eval}
        )
        ag_record = _write_dataset(
            data_root=data_root,
            name="agnews",
            train_rows=ag_train,
            eval_rows=ag_eval,
            eval_split="test",
            labels={0: "World", 1: "Sports", 2: "Business", 3: "Technology"},
            task="topic",
            source=f"ag_news@{AG_NEWS_REVISION}",
        )

    sst_record = _existing_dataset_record(data_root, "sst2")
    if sst_record is None:
        sst_train_path = _download_parquet(
            data_root, "glue", GLUE_REVISION, "sst2/train-00000-of-00001.parquet"
        )
        sst_validation_path = _download_parquet(
            data_root, "glue", GLUE_REVISION, "sst2/validation-00000-of-00001.parquet"
        )
        sst_train_raw = _parquet_rows(sst_train_path)
        sst_eval = [
            {"text": _normalize(row["sentence"]), "label": int(row["label"])}
            for row in _parquet_rows(sst_validation_path)
        ]
        sst_train = _unique_rows(
            sst_train_raw, "sentence", {row["text"] for row in sst_eval}
        )
        sst_record = _write_dataset(
            data_root=data_root,
            name="sst2",
            train_rows=sst_train,
            eval_rows=sst_eval,
            eval_split="validation",
            labels={0: "negative", 1: "positive"},
            task="sentiment",
            source=f"glue/sst2@{GLUE_REVISION}",
        )
    shutil.rmtree(data_root / "_downloads", ignore_errors=True)
    return {"agnews": ag_record, "sst2": sst_record}


def _model_record(model_dir: Path) -> dict:
    required = [model_dir / "config.json"]
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        weights = sorted(model_dir.glob("pytorch_model*.bin"))
    required.extend(weights)
    if len(required) < 2 or any(not path.is_file() for path in required):
        raise ValueError("frozen model snapshot is incomplete")
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in required
    }
    record = {"repo_id": LM_REPO, "files": files}
    record["model_sha256"] = _sha_bytes(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("APE_DATA_ROOT", "/data"))
    parser.add_argument("--model-src", default=os.environ.get("APE_MODEL_SRC"))
    parser.add_argument(
        "--reference-model",
        action="store_true",
        help="hash an existing image model through a symlink; use when only data is exported",
    )
    args = parser.parse_args()

    root = Path(args.data_root) / "prompt-optimization"
    model_dir = root / "models" / "Qwen2.5-0.5B-Instruct"
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=True)
    _snapshot(
        LM_REPO,
        model_dir,
        args.model_src,
        reference_model=args.reference_model,
    )
    datasets = build_data(data_dir)
    manifest = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "model": _model_record(model_dir),
        "datasets": datasets,
    }
    manifest["protocol_sha256"] = _sha_bytes(manifest)
    (data_dir / "manifest.json").write_bytes(_canonical_bytes(manifest))
    print(
        f"APE_PREPARE_DONE protocol={PROTOCOL} "
        f"protocol_sha256={manifest['protocol_sha256']} "
        f"agnews_eval={datasets['agnews']['eval_n']} "
        f"sst2_eval={datasets['sst2']['eval_n']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
