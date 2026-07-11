"""Prepare the pinned, official Flickr8k protocol for caption-* tasks.

The Hugging Face dataset revision used here exposes the standard Karpathy-style
6000/1000/1000 train/validation/test split.  We train on every image and all five
captions in the official train split, and score every image and all five references
in the official test split.  CLIP embeddings are cached once; the raw evaluation
images and references are not exposed to the evaluated agent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


TRAIN_SPLIT = "train"
EVAL_SPLIT = "test"
N_TRAIN = 6000
N_EVAL = 1000
REFS_PER_IMAGE = 5
CLIP_DIM = 512

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
FLICKR_DATASET_ID = "jxie/flickr8k"
FLICKR_REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
GPT2_MODEL_ID = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_FILE_SHA256 = {
    "model.safetensors": "c7d00560d8910fbed77ffad4065dee5011c41ba401b1064e749c498ba9e20373",
    "config.json": "50fda00afcbf90d2a7655c764fd8879f6ce8bed5624ff8231cae8889a7983cd4",
    "tokenizer.json": "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e",
}
CLIP_REPO_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CLIP_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
CLIP_CHECKPOINT = "open_clip_pytorch_model.bin"
CLIP_CHECKPOINT_SHA256 = "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"
SOURCE_MANIFEST = "source_manifest.json"
PROTOCOL = "flickr8k_official_v1"
MANIFEST_SCHEMA_VERSION = 3

# Stanford/Karpathy's canonical filename/split/caption artifact.  The pinned HF
# parquet intentionally omits filenames, so every row is mapped back to this
# source by its complete five-reference signature before it is accepted.
CANONICAL_ARCHIVE_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)
CANONICAL_ARCHIVE_SHA256 = (
    "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"
)
CANONICAL_JSON_MEMBER = "dataset_flickr8k.json"
CANONICAL_JSON_SHA256 = (
    "ce467057af54e8a8b7078fa6000c15cb3605dbfb36c3cc6a202cca90e8a9741e"
)
CANONICAL_FILENAME_SET_SHA256 = {
    "train": "fbb334d8b4d4bab05a65950cb0b8123079c40ba8d1c38d8aa360fa27459e8cf4",
    "test": "25d2fec0836bb4728d4672c46a5694dfbdb953a2ff5ba146f5ffaa7062512489",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_rgb_sha256(image) -> str:
    digest = hashlib.sha256()
    digest.update(f"RGB:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _captions_sha256(captions: list[str]) -> str:
    payload = json.dumps(captions, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalise_caption(caption: str) -> str:
    return " ".join(caption.strip().split())


def _caption_signature(captions: list[str]) -> tuple[str, ...]:
    return tuple(sorted(_normalise_caption(caption) for caption in captions))


def _caption_signature_sha256(captions: list[str]) -> str:
    payload = json.dumps(
        _caption_signature(captions),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _filename_set_sha256(filenames) -> str:
    payload = json.dumps(
        sorted(filenames), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _split_sha256(rows: list[dict]) -> str:
    payload = [
        {
            "source_filename": row["source_filename"],
            "decoded_rgb_sha256": row["decoded_rgb_sha256"],
            "captions_sha256": row["captions_sha256"],
            "canonical_captions_sha256": row["canonical_captions_sha256"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _load_split(split: str):
    from datasets import Dataset, concatenate_datasets, load_dataset

    datasets_cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "datasets",
        )
    )
    cached_root = (
        datasets_cache
        / "jxie___flickr8k"
        / "default"
        / "0.0.0"
        / FLICKR_REVISION
    )
    cached_arrows = sorted(cached_root.glob(f"flickr8k-{split}*.arrow"))
    if cached_arrows:
        shards = [Dataset.from_file(str(path)) for path in cached_arrows]
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        source = "pinned_arrow_cache"
    else:
        dataset = load_dataset(
            FLICKR_DATASET_ID,
            split=split,
            revision=FLICKR_REVISION,
        )
        source = "pinned_hf_revision"
    expected = N_TRAIN if split == TRAIN_SPLIT else N_EVAL
    if len(dataset) != expected:
        raise RuntimeError(
            f"official Flickr8k {split} split has {len(dataset)} rows, expected {expected}"
        )

    caption_columns = [f"caption_{index}" for index in range(REFS_PER_IMAGE)]
    required = {"image", *caption_columns}
    if not required <= set(dataset.column_names):
        raise RuntimeError(
            f"Flickr8k schema changed: missing {sorted(required - set(dataset.column_names))}"
        )

    print(
        f"FLICKR_SPLIT_LOADED split={split} rows={len(dataset)} "
        f"dataset_revision={FLICKR_REVISION} source={source}",
        flush=True,
    )
    return dataset


def _canonical_archive(cache_dir: Path) -> Path:
    override = os.environ.get("FLICKR8K_CANONICAL_ARCHIVE")
    target = Path(override) if override else cache_dir / "caption_datasets.zip"
    if not target.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="caption_datasets.", suffix=".zip", dir=cache_dir
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(CANONICAL_ARCHIVE_URL, timeout=120) as source:
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            if _sha256(temporary) != CANONICAL_ARCHIVE_SHA256:
                raise RuntimeError("canonical Flickr8k archive failed SHA-256 validation")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if _sha256(target) != CANONICAL_ARCHIVE_SHA256:
        raise RuntimeError("canonical Flickr8k archive failed SHA-256 validation")
    return target


def _load_canonical_index(cache_dir: Path) -> dict[tuple[str, ...], dict]:
    archive = _canonical_archive(cache_dir)
    with zipfile.ZipFile(archive) as handle:
        canonical_bytes = handle.read(CANONICAL_JSON_MEMBER)
    if hashlib.sha256(canonical_bytes).hexdigest() != CANONICAL_JSON_SHA256:
        raise RuntimeError("canonical Flickr8k JSON failed SHA-256 validation")
    canonical = json.loads(canonical_bytes)
    images = canonical.get("images")
    if canonical.get("dataset") != "flickr8k" or not isinstance(images, list):
        raise RuntimeError("canonical Flickr8k JSON has an invalid schema")
    counts = Counter(image.get("split") for image in images)
    if counts != {"train": 6000, "val": 1000, "test": 1000}:
        raise RuntimeError(f"canonical Flickr8k split counts changed: {counts}")

    index: dict[tuple[str, ...], dict] = {}
    filenames_by_split: dict[str, list[str]] = {"train": [], "test": []}
    for image in images:
        filename = image.get("filename")
        split = image.get("split")
        sentences = image.get("sentences")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".jpg")
            or split not in {"train", "val", "test"}
            or not isinstance(sentences, list)
            or len(sentences) != REFS_PER_IMAGE
        ):
            raise RuntimeError("canonical Flickr8k row has an invalid schema")
        captions = [sentence.get("raw") for sentence in sentences]
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise RuntimeError(f"canonical Flickr8k captions are invalid: {filename}")
        signature = _caption_signature(captions)
        if signature in index:
            raise RuntimeError("canonical Flickr8k five-caption signatures are not unique")
        index[signature] = {
            "source_filename": filename,
            "split": split,
            "canonical_captions_sha256": _caption_signature_sha256(captions),
        }
        if split in filenames_by_split:
            filenames_by_split[split].append(filename)
    if len(index) != 8000:
        raise RuntimeError(f"canonical Flickr8k index is incomplete: {len(index)}")
    observed_hashes = {
        split: _filename_set_sha256(filenames)
        for split, filenames in filenames_by_split.items()
    }
    if observed_hashes != CANONICAL_FILENAME_SET_SHA256:
        raise RuntimeError("canonical Flickr8k filename-set hashes changed")
    print(
        "FLICKR_CANONICAL_PROOF"
        f" archive_sha256={CANONICAL_ARCHIVE_SHA256}"
        f" json_sha256={CANONICAL_JSON_SHA256}"
        f" train_filename_set_sha256={observed_hashes['train']}"
        f" test_filename_set_sha256={observed_hashes['test']}"
        f" signatures={len(index)} collisions=0",
        flush=True,
    )
    return index


def _manifest_metadata() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "dataset_repo": FLICKR_DATASET_ID,
        "dataset_revision": FLICKR_REVISION,
        "train_split": TRAIN_SPLIT,
        "eval_split": EVAL_SPLIT,
        "train_count": N_TRAIN,
        "eval_count": N_EVAL,
        "references_per_image": REFS_PER_IMAGE,
        "gpt2_repo": GPT2_MODEL_ID,
        "gpt2_revision": GPT2_REVISION,
        "gpt2_file_sha256": GPT2_FILE_SHA256,
        "clip_model": CLIP_MODEL,
        "clip_pretrained_tag": CLIP_PRETRAINED,
        "clip_repo": CLIP_REPO_ID,
        "clip_revision": CLIP_REVISION,
        "clip_checkpoint": CLIP_CHECKPOINT,
        "clip_checkpoint_sha256": CLIP_CHECKPOINT_SHA256,
        "canonical_archive_url": CANONICAL_ARCHIVE_URL,
        "canonical_archive_sha256": CANONICAL_ARCHIVE_SHA256,
        "canonical_json_member": CANONICAL_JSON_MEMBER,
        "canonical_json_sha256": CANONICAL_JSON_SHA256,
        "canonical_filename_set_sha256": CANONICAL_FILENAME_SET_SHA256,
    }


def _prepared_hashes(out: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("train_clip.pt", "train_refs.json", "eval_clip.pt", "eval_refs.json"):
        hashes[name] = _sha256(out / name)
    for path in sorted((out / "gpt2").rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(out / "gpt2").parts:
            hashes[path.relative_to(out).as_posix()] = _sha256(path)
    return hashes


def _validate_source_rows(rows, *, split: str, count: int) -> bool:
    if not isinstance(rows, list) or len(rows) != count:
        return False
    observed_filenames: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        required = {
            "source_filename",
            "decoded_rgb_sha256",
            "captions_sha256",
            "canonical_captions_sha256",
        }
        if set(row) != required:
            return False
        if not all(
            isinstance(row[key], str) and len(row[key]) == 64
            for key in (
                "decoded_rgb_sha256",
                "captions_sha256",
                "canonical_captions_sha256",
            )
        ):
            return False
        filename = row["source_filename"]
        if not isinstance(filename, str) or not filename.endswith(".jpg"):
            return False
        observed_filenames.append(filename)
    return (
        len(set(observed_filenames)) == count
        and _filename_set_sha256(observed_filenames)
        == CANONICAL_FILENAME_SET_SHA256[split]
    )


def _prepared_is_valid(out: Path) -> bool:
    try:
        import torch

        manifest = json.loads((out / SOURCE_MANIFEST).read_text())
        if any(manifest.get(key) != value for key, value in _manifest_metadata().items()):
            return False
        selected = manifest.get("selected_sources")
        if not isinstance(selected, dict):
            return False
        train_sources = selected.get(TRAIN_SPLIT)
        eval_sources = selected.get(EVAL_SPLIT)
        if not _validate_source_rows(
            train_sources, split=TRAIN_SPLIT, count=N_TRAIN
        ) or not _validate_source_rows(eval_sources, split=EVAL_SPLIT, count=N_EVAL):
            return False
        if manifest.get("split_sha256") != {
            TRAIN_SPLIT: _split_sha256(train_sources),
            EVAL_SPLIT: _split_sha256(eval_sources),
        }:
            return False
        if manifest.get("canonical_filename_set_sha256") != {
            TRAIN_SPLIT: _filename_set_sha256(
                row["source_filename"] for row in train_sources
            ),
            EVAL_SPLIT: _filename_set_sha256(
                row["source_filename"] for row in eval_sources
            ),
        }:
            return False
        train_image_hashes = {row["decoded_rgb_sha256"] for row in train_sources}
        eval_image_hashes = {row["decoded_rgb_sha256"] for row in eval_sources}
        if train_image_hashes & eval_image_hashes:
            return False
        if manifest.get("prepared_sha256") != _prepared_hashes(out):
            return False

        train_emb = torch.load(
            out / "train_clip.pt", map_location="cpu", weights_only=True
        )
        eval_emb = torch.load(
            out / "eval_clip.pt", map_location="cpu", weights_only=True
        )
        train_refs = json.loads((out / "train_refs.json").read_text())
        eval_refs = json.loads((out / "eval_refs.json").read_text())
        valid_refs = lambda refs, count: (
            isinstance(refs, list)
            and len(refs) == count
            and all(
                isinstance(row, list)
                and len(row) == REFS_PER_IMAGE
                and all(isinstance(caption, str) and caption for caption in row)
                for row in refs
            )
        )
        if not valid_refs(train_refs, N_TRAIN) or not valid_refs(eval_refs, N_EVAL):
            return False
        for sources, references in (
            (train_sources, train_refs),
            (eval_sources, eval_refs),
        ):
            for source, captions in zip(sources, references):
                if source["captions_sha256"] != _captions_sha256(captions):
                    return False
                if source["canonical_captions_sha256"] != _caption_signature_sha256(
                    captions
                ):
                    return False
        return (
            tuple(train_emb.shape) == (N_TRAIN, CLIP_DIM)
            and tuple(eval_emb.shape) == (N_EVAL, CLIP_DIM)
            and train_emb.dtype == torch.float32
            and eval_emb.dtype == torch.float32
            and bool(torch.isfinite(train_emb).all())
            and bool(torch.isfinite(eval_emb).all())
            and (out / "gpt2" / "config.json").is_file()
        )
    except (OSError, EOFError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _source_record(
    image, captions: list[str], canonical: dict
) -> dict[str, str]:
    return {
        "source_filename": canonical["source_filename"],
        "decoded_rgb_sha256": _decoded_rgb_sha256(image),
        "captions_sha256": _captions_sha256(captions),
        "canonical_captions_sha256": canonical["canonical_captions_sha256"],
    }


def _build(data_root: Path) -> None:
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    out = data_root / "image-captioning"
    if _prepared_is_valid(out):
        print(f"CAPTION_DATA_VALID protocol={PROTOCOL} path={out}", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_workers = 1
    if device == "cpu":
        cpu_workers = int(os.environ.get("CAPTION_PREP_CPU_WORKERS", "1"))
        cpu_threads = int(os.environ.get("CAPTION_PREP_TORCH_THREADS", "1"))
        cpu_limit = os.cpu_count() or 1
        if not 1 <= cpu_workers <= cpu_limit:
            raise RuntimeError(f"CAPTION_PREP_CPU_WORKERS is invalid: {cpu_workers}")
        if not 1 <= cpu_threads <= cpu_limit:
            raise RuntimeError(
                f"CAPTION_PREP_TORCH_THREADS is invalid: {cpu_threads}"
            )
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        print(
            "CAPTION_CPU_THREADS"
            f" batch_workers={cpu_workers}"
            f" intraop={torch.get_num_threads()}"
            f" interop={torch.get_num_interop_threads()}",
            flush=True,
        )

    staging = out.with_name(f"{out.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    cache_dir = data_root / "_image-captioning-cache"
    canonical_index = _load_canonical_index(cache_dir)

    gpt_dir = staging / "gpt2"
    gpt_seed = os.environ.get("CAPTION_PREP_GPT2_SEED")
    if gpt_seed:
        seed_path = Path(gpt_seed)
        required_seed_files = {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        }
        if not seed_path.is_dir() or any(
            not (seed_path / name).is_file() for name in required_seed_files
        ):
            raise RuntimeError("pinned GPT-2 seed directory is incomplete")
        shutil.copytree(seed_path, gpt_dir)
        # Loading locally proves that the copied snapshot is compatible before
        # it is hashed into the immutable data manifest.
        tokenizer = GPT2TokenizerFast.from_pretrained(gpt_dir, local_files_only=True)
        gpt = GPT2LMHeadModel.from_pretrained(gpt_dir, local_files_only=True)
        del gpt
        print(f"CAPTION_GPT2_SEEDED source={seed_path}", flush=True)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            GPT2_MODEL_ID, revision=GPT2_REVISION
        )
        gpt = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_ID, revision=GPT2_REVISION)
        gpt_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(gpt_dir)
        gpt.save_pretrained(gpt_dir)
        del gpt

    for filename, expected_sha256 in GPT2_FILE_SHA256.items():
        path = gpt_dir / filename
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise RuntimeError(f"pinned GPT-2 file failed SHA-256 validation: {filename}")

    clip_checkpoint = Path(
        hf_hub_download(
            repo_id=CLIP_REPO_ID,
            filename=CLIP_CHECKPOINT,
            revision=CLIP_REVISION,
            cache_dir=str(cache_dir),
        )
    )
    if _sha256(clip_checkpoint) != CLIP_CHECKPOINT_SHA256:
        raise RuntimeError("pinned CLIP checkpoint failed SHA-256 validation")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=str(clip_checkpoint), cache_dir=str(cache_dir)
    )
    model = model.to(device).eval()

    train_rows = _load_split(TRAIN_SPLIT)
    eval_rows = _load_split(EVAL_SPLIT)

    @torch.no_grad()
    def embed_split(dataset, split: str, batch_size: int = 64):
        embeddings: list[torch.Tensor] = []
        references: list[list[str]] = []
        source_records: list[dict[str, str]] = []
        observed_filenames: set[str] = set()
        caption_columns = [f"caption_{index}" for index in range(REFS_PER_IMAGE)]

        @torch.no_grad()
        def embed_batch(start: int):
            stop = min(start + batch_size, len(dataset))
            raw_batch = dataset[start:stop]
            decoded_images = [image.convert("RGB") for image in raw_batch["image"]]
            batch_references = [
                [str(raw_batch[column][index]).strip() for column in caption_columns]
                for index in range(stop - start)
            ]
            canonical_rows = []
            for offset, captions in enumerate(batch_references):
                if any(not caption for caption in captions):
                    raise RuntimeError(f"invalid Flickr8k row {split}:{start + offset}")
                canonical = canonical_index.get(_caption_signature(captions))
                if canonical is None:
                    raise RuntimeError(
                        f"HF Flickr8k row has no canonical caption match: {split}:{start + offset}"
                    )
                if canonical["split"] != split:
                    raise RuntimeError(
                        f"HF Flickr8k row maps to canonical {canonical['split']} instead of {split}"
                    )
                canonical_rows.append(canonical)
            image_tensor = torch.stack(
                [preprocess(image) for image in decoded_images]
            ).to(device)
            encoded = model.encode_image(image_tensor).float().cpu()
            if not torch.isfinite(encoded).all():
                raise RuntimeError("CLIP produced non-finite embeddings")
            records = [
                _source_record(image, captions, canonical)
                for image, captions, canonical in zip(
                    decoded_images, batch_references, canonical_rows
                )
            ]
            return stop, encoded, batch_references, records

        starts = range(0, len(dataset), batch_size)
        if cpu_workers == 1:
            batch_results = map(embed_batch, starts)
            executor = None
        else:
            executor = ThreadPoolExecutor(
                max_workers=cpu_workers,
                thread_name_prefix="caption-clip",
            )
            batch_results = executor.map(embed_batch, starts)
        try:
            for stop, encoded, batch_references, records in batch_results:
                for record in records:
                    filename = record["source_filename"]
                    if filename in observed_filenames:
                        raise RuntimeError(
                            f"duplicate canonical Flickr8k filename: {filename}"
                        )
                    observed_filenames.add(filename)
                embeddings.append(encoded)
                references.extend(batch_references)
                source_records.extend(records)
                print(
                    f"FLICKR_CLIP_PROGRESS split={split} completed={stop}/{len(dataset)}",
                    flush=True,
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        result = torch.cat(embeddings, dim=0)
        if tuple(result.shape) != (len(dataset), CLIP_DIM):
            raise RuntimeError(f"unexpected CLIP shape {tuple(result.shape)}")
        if _filename_set_sha256(observed_filenames) != CANONICAL_FILENAME_SET_SHA256[split]:
            raise RuntimeError(f"HF Flickr8k {split} filename set is not canonical")
        return result, references, source_records

    train_emb, train_refs, train_sources = embed_split(train_rows, TRAIN_SPLIT)
    eval_emb, eval_refs, eval_sources = embed_split(eval_rows, EVAL_SPLIT)
    torch.save(train_emb, staging / "train_clip.pt")
    torch.save(eval_emb, staging / "eval_clip.pt")
    (staging / "train_refs.json").write_text(
        json.dumps(train_refs, ensure_ascii=False, separators=(",", ":"))
    )
    (staging / "eval_refs.json").write_text(
        json.dumps(eval_refs, ensure_ascii=False, separators=(",", ":"))
    )

    manifest = _manifest_metadata()
    manifest["selected_sources"] = {
        TRAIN_SPLIT: train_sources,
        EVAL_SPLIT: eval_sources,
    }
    manifest["split_sha256"] = {
        TRAIN_SPLIT: _split_sha256(train_sources),
        EVAL_SPLIT: _split_sha256(eval_sources),
    }
    manifest["prepared_sha256"] = _prepared_hashes(staging)
    (staging / SOURCE_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    if not _prepared_is_valid(staging):
        raise RuntimeError("prepared official Flickr8k data failed post-build validation")
    if out.exists():
        shutil.rmtree(out)
    staging.replace(out)
    print(
        f"CAPTION_DATA_BUILT protocol={PROTOCOL} train_images={N_TRAIN} "
        f"train_pairs={N_TRAIN * REFS_PER_IMAGE} eval_images={N_EVAL} "
        f"clip_dim={CLIP_DIM}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    _build(Path(args.data_root))


if __name__ == "__main__":
    main()
