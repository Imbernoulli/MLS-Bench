"""Trusted pipeline for the machine-translation (mt-*) MLS-Bench tasks.

Every task performs inference with one pinned Helsinki-NLP MarianMT checkpoint,
translates all 2,000 rows of the matching pinned OPUS-100 test split, and scores
corpus sacreBLEU and chrF.  Data, tokenizer, checkpoint, and model provenance are
validated here before any metric record can be emitted.

Three DIRECTIONS / settings (all -> English so the sacreBLEU reference tokenizer
and metric are consistent across settings):

  * de_en  : German  -> English   (Helsinki-NLP/opus-mt-de-en)
  * fr_en  : French  -> English   (Helsinki-NLP/opus-mt-fr-en)
  * ru_en  : Russian -> English   (Helsinki-NLP/opus-mt-ru-en)

The agent-editable surface never touches the model, the corpus, the references,
or the sacreBLEU evaluator. It controls ONLY the one DECODING component the task
is about (beam / length-normalization / repetition-block / coverage / sampling /
temperature / n-best rerank / tokenization+truncation / batch+max-length /
early-stopping / detok / decode strategy).

Why sacreBLEU (not a hand-rolled BLEU): sacreBLEU is the community-standard,
reproducible, tokenization-controlled corpus BLEU (Post 2018), so scores are
comparable and non-gameable. BLEU is a geometric mean of n-gram precisions with
a brevity penalty, so an empty output scores 0 and a copy-the-source (wrong
language) output scores ~0 — a real translation cannot be beaten by degenerate
tricks.

Everything runs offline.  Unknown or mismatched artifacts are fatal; there is no
runtime download, model substitution, CPU fallback, or score fallback.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL mt-* tasks / directions)
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = "mt-opus100-provenance-v2"
DATASET_ID = "Helsinki-NLP/opus-100"
DATASET_REVISION = "805090dc28bf78897da9641cdf08b61287580df9"
OFFICIAL_TEST_PAIRS = 2000  # complete OPUS-100 test split for every direction
MAX_INPUT_TOKENS = 128     # source truncation (OPUS-100 sentences are ~15-25 words)
MAX_NEW_TOKENS_CAP = 160   # hard cap on generated length
GEN_BATCH_SIZE = 16        # default generation batch size (a task may override)
SEED = 42

# The three directions this repo evaluates over. Each entry: (model_dir, test_file).
# All target English so the sacreBLEU reference side is one language.
DIRECTIONS = {
    "de_en": ("opus-mt-de-en", "de_en_test.jsonl"),
    "fr_en": ("opus-mt-fr-en", "fr_en_test.jsonl"),
    "ru_en": ("opus-mt-ru-en", "ru_en_test.jsonl"),
}

DATA_SPECS = {
    "de_en": {
        "source_language": "de",
        "source_path": "de-en/test-00000-of-00001.parquet",
        "source_sha256": "05913515e9dc8c11bc03570bd00ae5b551c32b03e07901369f91372ad63a3f11",
        "output_sha256": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
    },
    "fr_en": {
        "source_language": "fr",
        "source_path": "en-fr/test-00000-of-00001.parquet",
        "source_sha256": "6e5862c14744efb89cf4c807cf0fd1a5969249935f21a1d03f3fbdbc0fb81971",
        "output_sha256": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
    },
    "ru_en": {
        "source_language": "ru",
        "source_path": "en-ru/test-00000-of-00001.parquet",
        "source_sha256": "96bf7751ebd69615e1377a06cf49bb7d2d153124c77764620de435d0afc71935",
        "output_sha256": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
    },
}

# File digests and count-probe details are recorded in artifact_provenance.json.
# Counts were first derived from the safetensors header (fr-en) or legacy torch
# pickle metadata (de-en/ru-en), then confirmed by an actual model load on the
# pinned repository image.  runtime_probe.json records the worker evidence.
MODEL_SPECS = {
    "de_en": {
        "repository": "Helsinki-NLP/opus-mt-de-en",
        "revision": "1a922f3b32a8e809e17a47d4b32142d8105924e5",
        "checkpoint_file": "pytorch_model.bin",
        "checkpoint_format": "pytorch",
        "checkpoint_tensor_elements": 74_468_597,
        "parameter_count": 74_410_496,
        "vocab_size": 58_101,
        "files": {
            "config.json": (1381, "89368ef76ea89581025cdf605caac75b8a22af2c1a90ec57c8a5001f10537eeb"),
            "generation_config.json": (293, "e5cccf365761fb31175fc978a75f5e857a822f173d0e447aa3a25d7cbcda15a4"),
            "pytorch_model.bin": (297_928_209, "e743c3070f61f477cb62fe95ef2c9be2e77f3e488cb6b8030ff8a19e8295c87d"),
            "source.spm": (796_845, "bbd1f495eea99c8e21ae086d9146e0fa7b096c3dfdd9ba07ab8b631889df5c9b"),
            "target.spm": (768_489, "678f2a1177d8389f67b66299762dcc4fc567e89b07e212ba91b0c56daecf47ce"),
            "tokenizer_config.json": (42, "51c3c3260d27cb7c4d11d0c53752b8fe87f2367129d7636fa917ce588b97306c"),
            "vocab.json": (1_273_232, "0d70d89fee4a8b4ef99a56d712163baadcabd5600a597f71515547ee70306329"),
        },
    },
    "fr_en": {
        "repository": "Helsinki-NLP/opus-mt-fr-en",
        "revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
        "checkpoint_file": "model.safetensors",
        "checkpoint_format": "safetensors",
        "checkpoint_tensor_elements": 75_193_466,
        "parameter_count": 75_133_952,
        "vocab_size": 59_514,
        "files": {
            "config.json": (1416, "b3be13d046d9899d7aab8cf4ed624d9a79f5776038ba793f6b4d2ce3e02192f7"),
            "generation_config.json": (293, "4956fb9a7caaad7579cf8bb789c1e578b8a1cf48a0a8b779fda2f95dd10bbaa5"),
            "model.safetensors": (300_803_608, "6e3837f34b903802c3d0d670362b997cee6e87584a1108eb3fa89e4625e4424a"),
            "source.spm": (802_397, "78d0e717c77053f1c4b856d8661d9cb87c64f083a35418c087b9146300e4f585"),
            "target.spm": (778_395, "173e9f493a668fe396d599e28d414a201193094e6ffd7a4678e5aab0f6d3d838"),
            "tokenizer_config.json": (42, "47de9ce87378593016432f8dc657202c03913ab3ce0c15d7f78d51edfc3ff9a3"),
            "vocab.json": (1_339_166, "945c604346ce15ce4aff9001001e7f925e336d942c4087017f191871162cbdc4"),
        },
    },
    "ru_en": {
        "repository": "Helsinki-NLP/opus-mt-ru-en",
        "revision": "fbd6dc73284f95536648512cc21d57f19191961a",
        "checkpoint_file": "pytorch_model.bin",
        "checkpoint_format": "pytorch",
        "checkpoint_tensor_elements": 76_734_518,
        "parameter_count": 76_672_000,
        "vocab_size": 62_518,
        "files": {
            "config.json": (1381, "5ea76c78596bce8fe005ef89e00de0924bf83e5f532ce08784ff1fcefb699f5f"),
            "generation_config.json": (293, "31cff5e74efc263ff53efac09e3e350cd462d5c8198b1136b455d63a02d8cad5"),
            "pytorch_model.bin": (306_991_893, "535450eb5613f3cc912f9ca3e54cfef6c14d201b319c24a88faf776a65538b5d"),
            "source.spm": (1_080_169, "745998e51ba5b058e38b7ac7765c25c43ed5c1c39cc92b27163b9b2e323c9d7c"),
            "target.spm": (802_781, "16bebef1389a0b8ab452772c4e35b9e605e5713f8ac7baa71ca701394eaa086d"),
            "tokenizer_config.json": (42, "8d826099b8c67179c83ab4ff94aff7ca7bf24ca14319cb0287b9a0b4c40b2a96"),
            "vocab.json": (2_601_758, "33e95da3be3fa3b50169c4c46693ba2f29fbf4cb29d99044bd07d72d181fa1e9"),
        },
    },
}

TASK_SURFACES = {
    "mt-batch-maxlen": "build_max_new_tokens",
    "mt-decoding-beam": "build_beam_config",
    "mt-decoding-strategy": "build_strategy",
    "mt-decoding-temperature": "build_temperature",
    "mt-diverse-beam": "build_divbeam_config",
    "mt-early-stopping": "build_early_stopping",
    "mt-length-penalty": "build_length_config",
    "mt-no-repeat-ngram": "build_norep_config",
    "mt-postprocess-detok": "build_postproc",
    "mt-repetition-penalty": "build_reppen_config",
    "mt-sampling-vs-beam": "build_mode",
    "mt-tokenization-truncation": "build_source_max_tokens",
}

TGT_LANG = "en"


def _data_root() -> Path:
    return Path(os.environ.get("MT_DATA", "/data/machine-translation/data"))


def _models_root() -> Path:
    # MT_MODEL historically pointed at the single de-en model dir; derive the
    # models root from it so per-direction models resolve next to it.
    mm = os.environ.get("MT_MODEL")
    if mm:
        p = Path(mm)
        # if MT_MODEL is .../models/opus-mt-de-en, the root is its parent
        if p.name.startswith("opus-mt-"):
            return p.parent
        return p
    return Path("/data/machine-translation/models")


def direction() -> str:
    """Active direction key (env MT_DIR; default de_en)."""
    d = os.environ.get("MT_DIR", "de_en")
    if d not in DIRECTIONS:
        raise SystemExit(f"MT_DIR={d!r} not one of {sorted(DIRECTIONS)}")
    return d


def model_path(dir_key: str | None = None) -> str:
    """FROZEN small pretrained OPUS-MT model for the active direction."""
    dk = dir_key or direction()
    model_dir, _ = DIRECTIONS[dk]
    # de_en can also be resolved from the legacy MT_MODEL env directly.
    if dk == "de_en":
        mm = os.environ.get("MT_MODEL")
        if mm and Path(mm).name.startswith("opus-mt-de-en"):
            return mm
    return str(_models_root() / model_dir)


def data_root() -> Path:
    return _data_root()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_source_manifest() -> dict:
    splits = []
    for dk, (_model_dir, output_file) in DIRECTIONS.items():
        spec = DATA_SPECS[dk]
        splits.append({
            "direction": dk,
            "rows": OFFICIAL_TEST_PAIRS,
            "source_language": spec["source_language"],
            "target_language": TGT_LANG,
            "source_path": spec["source_path"],
            "source_sha256": spec["source_sha256"],
            "output_file": output_file,
            "output_sha256": spec["output_sha256"],
        })
    return {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "expected_rows_per_direction": OFFICIAL_TEST_PAIRS,
        "splits": splits,
    }


def _file_record(name: str, spec: dict) -> dict:
    size, digest = spec["files"][name]
    return {"path": name, "size": size, "sha256": digest}


def expected_tokenizer_manifest(dir_key: str) -> dict:
    spec = MODEL_SPECS[dir_key]
    names = ("source.spm", "target.spm", "tokenizer_config.json", "vocab.json")
    return {
        "schema_version": 1,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "vocab_size": spec["vocab_size"],
        "files": [_file_record(name, spec) for name in names],
    }


def expected_model_manifest(dir_key: str) -> dict:
    spec = MODEL_SPECS[dir_key]
    tokenizer = expected_tokenizer_manifest(dir_key)
    tokenizer_digest = hashlib.sha256(_canonical_json_bytes(tokenizer)).hexdigest()
    return {
        "schema_version": 1,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "parameter_count": spec["parameter_count"],
        "checkpoint_tensor_elements": spec["checkpoint_tensor_elements"],
        "checkpoint": {
            **_file_record(spec["checkpoint_file"], spec),
            "format": spec["checkpoint_format"],
        },
        "model_files": [
            _file_record("config.json", spec),
            _file_record("generation_config.json", spec),
        ],
        "tokenizer": tokenizer,
        "tokenizer_manifest_sha256": tokenizer_digest,
    }


def source_manifest_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(expected_source_manifest())).hexdigest()


def model_manifest_sha256(dir_key: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(expected_model_manifest(dir_key))
    ).hexdigest()


def setup(seed: int = SEED):
    """Pin the scored seed and require the scheduler-provided CUDA device."""
    import random

    import numpy as np
    import torch

    if seed != SEED:
        raise ValueError(f"machine-translation protocol requires seed {SEED}, got {seed}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("machine-translation verification requires one visible CUDA GPU")
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda:0")


def emit_protocol(task_name: str, surface_name: str, seed: int) -> None:
    expected_surface = TASK_SURFACES.get(task_name)
    if expected_surface != surface_name:
        raise ValueError(
            f"invalid machine-translation task/surface binding: "
            f"{task_name!r}/{surface_name!r}"
        )
    if seed != SEED:
        raise ValueError(f"protocol seed mismatch: expected {SEED}, got {seed}")
    print(
        f"MT_PROTOCOL version={PROTOCOL_VERSION} task={task_name} "
        f"surface={surface_name} direction={direction()} seed={seed}",
        flush=True,
    )


def _verify_source_manifest() -> dict:
    root = _data_root()
    expected = expected_source_manifest()
    expected_bytes = _canonical_json_bytes(expected)
    manifest_path = root / "source_manifest.json"
    try:
        actual_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read OPUS source manifest: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise ValueError("staged OPUS source manifest does not match the pinned protocol")
    expected_names = {record["output_file"] for record in expected["splits"]}
    expected_names.add("source_manifest.json")
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"unexpected staged OPUS data inventory: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )
    return {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "manifest_sha256": hashlib.sha256(actual_bytes).hexdigest(),
    }


def load_dataset(dir_key: str | None = None):
    """Load the complete official OPUS-100 test split for one direction."""
    dk = dir_key or direction()
    if dk not in DIRECTIONS:
        raise ValueError(f"unknown direction {dk!r}")
    proof = _verify_source_manifest()
    _, fname = DIRECTIONS[dk]
    fp = _data_root() / fname
    try:
        actual_sha256 = _sha256_file(fp)
    except OSError as exc:
        raise FileNotFoundError(f"cannot read staged OPUS-100 split {fp}: {exc}") from exc
    if actual_sha256 != DATA_SPECS[dk]["output_sha256"]:
        raise ValueError(
            f"official OPUS-100 split digest mismatch for {fp}: expected "
            f"{DATA_SPECS[dk]['output_sha256']}, got {actual_sha256}"
        )
    srcs: List[str] = []
    refs: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {fp}:{line_number}")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {fp}:{line_number}: {exc}") from exc
            if not isinstance(rec, dict) or set(rec) != {"src", "ref"}:
                raise ValueError(
                    f"invalid record schema at {fp}:{line_number}; expected src/ref"
                )
            src, ref = rec["src"], rec["ref"]
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"empty source at {fp}:{line_number}")
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"empty reference at {fp}:{line_number}")
            srcs.append(src)
            refs.append(ref)
    if len(srcs) != OFFICIAL_TEST_PAIRS:
        raise ValueError(
            f"incomplete official split {fp}: expected {OFFICIAL_TEST_PAIRS} rows, "
            f"got {len(srcs)}"
        )
    proof.update({
        "direction": dk,
        "split_sha256": actual_sha256,
        "rows": len(srcs),
    })
    return srcs, refs, proof


def _verify_model_artifacts(dir_key: str) -> dict:
    spec = MODEL_SPECS[dir_key]
    model_dir = Path(model_path(dir_key))
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise FileNotFoundError(f"missing staged OPUS-MT model directory: {model_dir}")
    expected_manifest = expected_model_manifest(dir_key)
    expected_manifest_bytes = _canonical_json_bytes(expected_manifest)
    manifest_path = model_dir / "model_manifest.json"
    try:
        actual_manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read staged model manifest: {exc}") from exc
    if actual_manifest_bytes != expected_manifest_bytes:
        raise ValueError(f"model manifest mismatch for {dir_key}")

    expected_names = set(spec["files"]) | {"model_manifest.json"}
    actual_names = {path.name for path in model_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"unexpected model inventory for {dir_key}: expected "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )
    for name, (expected_size, expected_digest) in spec["files"].items():
        path = model_dir / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing regular model artifact: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"model artifact size mismatch for {path}: expected "
                f"{expected_size}, got {actual_size}"
            )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"model artifact digest mismatch for {path}: expected "
                f"{expected_digest}, got {actual_digest}"
            )
    tokenizer_digest = hashlib.sha256(
        _canonical_json_bytes(expected_tokenizer_manifest(dir_key))
    ).hexdigest()
    return {
        "direction": dir_key,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "manifest_sha256": hashlib.sha256(actual_manifest_bytes).hexdigest(),
        "tokenizer_manifest_sha256": tokenizer_digest,
        "checkpoint_sha256": spec["files"][spec["checkpoint_file"]][1],
        "parameter_count": spec["parameter_count"],
    }


def load_model_and_tokenizer(device, dir_key: str | None = None):
    """FROZEN OPUS-MT MarianMT for the active direction, eval mode, offline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    dk = dir_key or direction()
    if dk not in MODEL_SPECS:
        raise ValueError(f"unknown model direction {dk!r}")
    proof = _verify_model_artifacts(dk)
    spec = MODEL_SPECS[dk]
    mp = model_path(dk)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
    model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
        mp,
        local_files_only=True,
        torch_dtype=torch.float32,
        use_safetensors=spec["checkpoint_format"] == "safetensors",
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise RuntimeError(f"checkpoint loading reported {key}: {loading_info[key]}")
    if model.__class__.__name__ != "MarianMTModel":
        raise TypeError(f"expected MarianMTModel, got {model.__class__.__name__}")
    if tok.__class__.__name__ != "MarianTokenizer":
        raise TypeError(f"expected MarianTokenizer, got {tok.__class__.__name__}")
    if int(model.config.vocab_size) != spec["vocab_size"] or len(tok) != spec["vocab_size"]:
        raise ValueError("loaded model/tokenizer vocabulary does not match pinned metadata")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec["parameter_count"]:
        raise ValueError(
            f"loaded parameter count mismatch for {dk}: expected "
            f"{spec['parameter_count']}, got {parameter_count}"
        )
    final_bias = getattr(model, "final_logits_bias", None)
    tensor_elements = parameter_count + (final_bias.numel() if final_bias is not None else 0)
    if tensor_elements != spec["checkpoint_tensor_elements"]:
        raise ValueError(
            f"loaded checkpoint tensor count mismatch for {dk}: expected "
            f"{spec['checkpoint_tensor_elements']}, got {tensor_elements}"
        )
    model.to(device)
    model.eval()
    return model, tok, proof


def emit_provenance(model_proof: dict, data_proof: dict) -> None:
    dk = direction()
    if model_proof.get("direction") != dk or data_proof.get("direction") != dk:
        raise ValueError("provenance direction does not match the active setting")
    print(
        f"MT_MODEL direction={dk} repository={model_proof['repository']} "
        f"revision={model_proof['revision']} "
        f"manifest_sha256={model_proof['manifest_sha256']} "
        f"tokenizer_manifest_sha256={model_proof['tokenizer_manifest_sha256']} "
        f"checkpoint_sha256={model_proof['checkpoint_sha256']} "
        f"parameters={model_proof['parameter_count']}",
        flush=True,
    )
    print(
        f"MT_DATA direction={dk} dataset={data_proof['dataset']} "
        f"revision={data_proof['revision']} "
        f"manifest_sha256={data_proof['manifest_sha256']} "
        f"split_sha256={data_proof['split_sha256']} rows={data_proof['rows']}",
        flush=True,
    )


def emit_result(task_name: str, surface_name: str, scores: dict,
                pred_len_words: float, elapsed: float, rows: int) -> None:
    expected_surface = TASK_SURFACES.get(task_name)
    if expected_surface != surface_name:
        raise ValueError("result task/surface binding is invalid")
    bleu = float(scores["bleu"])
    chrf = float(scores["chrf"])
    if (
        rows != OFFICIAL_TEST_PAIRS
        or not math.isfinite(bleu)
        or not math.isfinite(chrf)
        or not math.isfinite(pred_len_words)
        or not math.isfinite(elapsed)
        or not 0.0 <= bleu <= 100.0
        or not 0.0 <= chrf <= 100.0
        or pred_len_words < 0.0
        or elapsed <= 0.0
    ):
        raise ValueError("refusing to emit an invalid machine-translation result")
    dk = direction()
    print(
        f"MT_METRICS task={task_name} surface={surface_name} direction={dk} "
        f"bleu={bleu:.6f} chrf={chrf:.6f} n_pairs={rows} "
        f"plen={pred_len_words:.6f} elapsed={elapsed:.6f}",
        flush=True,
    )
    print(
        f"MT_COMPLETE task={task_name} surface={surface_name} "
        f"direction={dk} status=ok",
        flush=True,
    )


_SURFACE_SOURCE_BYTES = 64 * 1024
_SURFACE_AST_NODES = 256
_SURFACE_NAMES = {
    "build_beam_config",
    "build_divbeam_config",
    "build_early_stopping",
    "build_length_config",
    "build_max_new_tokens",
    "build_norep_config",
    "build_postproc",
    "build_reppen_config",
    "build_mode",
    "build_strategy",
    "build_temperature",
    "build_source_max_tokens",
}


def _surface_error(message: str) -> ValueError:
    return ValueError(f"unsafe machine-translation configuration: {message}")


def _validate_data_literal(value, *, path: str = "return") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _surface_error(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_data_literal(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _surface_error(f"{path} has a non-string mapping key")
            _validate_data_literal(item, path=f"{path}[{key!r}]")
        return
    raise _surface_error(f"{path} has unsupported type {type(value).__name__}")


def load_surface_value(sol_path: str, attr: str):
    """Read one zero-argument literal-return surface without executing the file."""
    if attr not in _SURFACE_NAMES:
        raise _surface_error(f"unsupported surface name {attr!r}")
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _surface_error(f"cannot read solution: {exc}") from exc
    if len(source.encode("utf-8")) > _SURFACE_SOURCE_BYTES:
        raise _surface_error("source exceeds 64 KiB")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise _surface_error(f"source does not parse: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _SURFACE_AST_NODES:
        raise _surface_error("AST exceeds 256 nodes")

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == attr
    ]
    if len(functions) != 1:
        raise _surface_error(f"solution must define exactly one {attr}")
    function = functions[0]
    args = function.args
    if (
        function.decorator_list
        or args.posonlyargs
        or args.args
        or args.vararg is not None
        or args.kwonlyargs
        or args.kwarg is not None
        or args.defaults
        or args.kw_defaults
    ):
        raise _surface_error(f"{attr} must be undecorated and accept no arguments")
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise _surface_error(f"{attr} body must contain exactly one return statement")

    for node in tree.body:
        if node is function:
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and [alias.name for alias in node.names] == ["annotations"]
        ):
            continue
        raise _surface_error("top-level executable statements and imports are forbidden")

    try:
        value = ast.literal_eval(function.body[0].value)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise _surface_error(f"{attr} must return one literal value") from exc
    _validate_data_literal(value)
    return value


def require_config(value, attr: str, required: set[str], optional: set[str] | None = None):
    """Validate an editable mapping without supplying verifier-side defaults."""
    if not isinstance(value, dict):
        raise TypeError(f"{attr} must return a mapping, got {type(value).__name__}")
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing:
        raise ValueError(f"{attr} is missing required keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{attr} has unsupported keys: {sorted(unexpected)}")
    return value


def require_int(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}], got {value}")
    return value


def require_real(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum},{maximum}], got {value}")
    return result


# ---------------------------------------------------------------------------
# Generation config sanitisation
# ---------------------------------------------------------------------------
def _sanitize_gen_kwargs(gen_kwargs: dict) -> dict:
    """Validate the agent's decode config: only known generation knobs allowed,
    and hard caps so nothing can blow up the compute budget or game the metric."""
    allowed = {
        "num_beams", "min_length", "max_new_tokens", "length_penalty",
        "no_repeat_ngram_size", "repetition_penalty", "early_stopping",
        "do_sample", "temperature", "top_k", "top_p", "num_beam_groups",
        "diversity_penalty", "num_return_sequences",
    }
    if not isinstance(gen_kwargs, dict):
        raise TypeError("generation configuration must be a mapping")
    out = {}
    for k, v in gen_kwargs.items():
        if k not in allowed:
            raise SystemExit(
                f"decode kwarg {k!r} not allowed; permitted: {sorted(allowed)}"
            )
        out[k] = v
    out["num_beams"] = require_int(out.get("num_beams", 1), "num_beams", 1, 12)
    out["max_new_tokens"] = require_int(
        out.get("max_new_tokens", MAX_NEW_TOKENS_CAP),
        "max_new_tokens",
        1,
        MAX_NEW_TOKENS_CAP,
    )
    out["min_length"] = require_int(
        out.get("min_length", 0), "min_length", 0, MAX_NEW_TOKENS_CAP
    )
    if out["min_length"] > out["max_new_tokens"]:
        raise ValueError("min_length cannot exceed max_new_tokens")
    if "length_penalty" in out:
        out["length_penalty"] = require_real(
            out["length_penalty"], "length_penalty", 0.0, 5.0
        )
    if "no_repeat_ngram_size" in out:
        out["no_repeat_ngram_size"] = require_int(
            out["no_repeat_ngram_size"], "no_repeat_ngram_size", 0, 10
        )
    if "repetition_penalty" in out:
        out["repetition_penalty"] = require_real(
            out["repetition_penalty"], "repetition_penalty", 0.1, 5.0
        )
    if "temperature" in out:
        out["temperature"] = require_real(
            out["temperature"], "temperature", 0.05, 5.0
        )
    if "top_k" in out:
        out["top_k"] = require_int(out["top_k"], "top_k", 0, 1000)
    if "top_p" in out:
        out["top_p"] = require_real(out["top_p"], "top_p", 0.0, 1.0)
    if "diversity_penalty" in out:
        out["diversity_penalty"] = require_real(
            out["diversity_penalty"], "diversity_penalty", 0.0, 5.0
        )
    if "do_sample" in out and not isinstance(out["do_sample"], bool):
        raise TypeError("do_sample must be a boolean")
    if "early_stopping" in out and out["early_stopping"] not in (True, False, "never"):
        raise ValueError("early_stopping must be True, False, or 'never'")
    if "num_return_sequences" in out:
        out["num_return_sequences"] = require_int(
            out["num_return_sequences"],
            "num_return_sequences",
            1,
            out["num_beams"],
        )
    if "num_beam_groups" in out:
        out["num_beam_groups"] = require_int(
            out["num_beam_groups"], "num_beam_groups", 1, out["num_beams"]
        )
        if out["num_beams"] % out["num_beam_groups"]:
            raise ValueError("num_beam_groups must divide num_beams")
    return out


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------
def translate(model, tok, sources: List[str], gen_kwargs: dict, device,
              max_input_tokens: int = MAX_INPUT_TOKENS,
              batch_size: int = GEN_BATCH_SIZE) -> List[str]:
    """Decode a translation for every source sentence with the agent's config.

    The frozen model is fixed; only `gen_kwargs`, the source truncation length
    (`max_input_tokens`) and the batch size may vary (some tasks expose those).
    """
    import torch

    if not sources:
        raise ValueError("translation source list is empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not 1 <= max_input_tokens <= MAX_INPUT_TOKENS
    ):
        raise ValueError(
            f"max_input_tokens must be an integer in [1,{MAX_INPUT_TOKENS}], "
            f"got {max_input_tokens!r}"
        )
    gk = _sanitize_gen_kwargs(gen_kwargs)
    preds: List[str] = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        enc = tok(
            batch,
            max_length=max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **gk,
            )
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        if len(decoded) != len(batch) or any(not isinstance(item, str) for item in decoded):
            raise RuntimeError(
                f"decoder returned {len(decoded)} outputs for a batch of {len(batch)} sources"
            )
        preds.extend(decoded)
    if len(preds) != len(sources):
        raise RuntimeError(
            f"decoder returned {len(preds)} outputs for {len(sources)} sources"
        )
    return preds


def translate_nbest(model, tok, sources: List[str], gen_kwargs: dict, device,
                    n: int, max_input_tokens: int = MAX_INPUT_TOKENS,
                    batch_size: int = GEN_BATCH_SIZE) -> List[List[str]]:
    """Return the top-`n` beam hypotheses per source (for n-best reranking tasks).

    Each output row is a list of `n` candidate strings (best-first by model
    score). `num_beams` must be >= n; the harness sets num_return_sequences=n.
    """
    import torch

    n = require_int(n, "n", 1, 12)
    if not sources:
        raise ValueError("translation source list is empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not 1 <= max_input_tokens <= MAX_INPUT_TOKENS
    ):
        raise ValueError(
            f"max_input_tokens must be an integer in [1,{MAX_INPUT_TOKENS}], "
            f"got {max_input_tokens!r}"
        )
    gk = _sanitize_gen_kwargs(gen_kwargs)
    gk["num_beams"] = max(gk.get("num_beams", 1), n)
    gk["num_return_sequences"] = n
    gk.pop("do_sample", None)          # n-best is a beam concept
    all_lists: List[List[str]] = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        enc = tok(batch, max_length=max_input_tokens, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"], **gk)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        if len(decoded) != len(batch) * n:
            raise RuntimeError(
                f"n-best decoder returned {len(decoded)} outputs for "
                f"{len(batch)} sources with n={n}"
            )
        # out is (batch*n) rows, grouped per source
        for b in range(len(batch)):
            all_lists.append(decoded[b * n:(b + 1) * n])
    if len(all_lists) != len(sources) or any(len(row) != n for row in all_lists):
        raise RuntimeError("n-best decoder output is incomplete")
    return all_lists


def score_bleu_chrf(preds: List[str], refs: List[str]) -> dict:
    """Corpus sacreBLEU + chrF over the slice via the `sacrebleu` library
    (Post 2018; the community-standard, tokenization-controlled, non-gameable
    corpus BLEU). Single reference -> wrap the ref list once: [refs].

    Returns {"bleu": <0-100>, "chrf": <0-100>}.
    """
    import sacrebleu

    if not preds or len(preds) != len(refs):
        raise ValueError(
            f"predictions/references must be non-empty and aligned, got "
            f"{len(preds)} and {len(refs)}"
        )
    if any(not isinstance(item, str) for item in preds + refs):
        raise TypeError("predictions and references must contain only strings")

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    if not math.isfinite(bleu) or not math.isfinite(chrf):
        raise ValueError(f"non-finite translation metrics: bleu={bleu}, chrf={chrf}")
    return {"bleu": float(bleu), "chrf": float(chrf)}


def mean_pred_len_words(preds: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(len(p.split()) for p in preds) / len(preds)
